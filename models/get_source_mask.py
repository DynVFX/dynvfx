import os
import sys
sys.path.append("third_party/evfsam2")
from pathlib import Path
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from torchvision.transforms import ToTensor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import write_video


video_codec = "libx264"
video_options = {
    "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
    "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
}

class SourceMaskExtractor:
    def __init__(self, datat_path, output_path, precision, object_of_interest_dict, image_size=224, id_for_sam="first_frame",
                 semantic_level=False):
        self.datat_path = datat_path
        self.image_size = image_size
        self.object_of_interest_dict = object_of_interest_dict
        self.semantic_level = semantic_level
        self.id_for_sam = id_for_sam
        if self.id_for_sam == "all_frames":
            self.id_list = [j for j in range(49)]
        elif self.id_for_sam == "3_key_frames":
            self.id_list = [0, 24, 48]
        elif self.id_for_sam == "5_key_frames":
            self.id_list = [0, 12, 24, 36, 48]
        elif self.id_for_sam == "first_frame":
            self.id_list = [0]
        self.output_path = output_path
        if precision == "bf16":
            self.precision = torch.bfloat16
        elif precision == "fp16":
            self.precision = torch.float16
        self.resolution = (720, 480)

        self.tokenizer, self.model = self.init_models(version='YxZhang/evf-sam2-multitask')

    def init_models(self, version='YxZhang/evf-sam2-multitask'):
        tokenizer = AutoTokenizer.from_pretrained(
            version,
            padding_side="right",
            use_fast=False,
        )

        kwargs = {"torch_dtype": self.precision}

        from model.evf_sam2_video import EvfSam2Model
        model = EvfSam2Model.from_pretrained(
            version, low_cpu_mem_usage=True, **kwargs
        )
        model = model.cuda()
        model.eval()
        return tokenizer, model

    def beit3_preprocess(self, x):
        '''
        preprocess for BEIT-3 model.
        input: ndarray
        output: torch.Tensor
        '''
        beit_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return beit_preprocess(x)

    def return_mask_img(self, mask):
        im = (mask * 255.0).round().astype("uint8")
        return Image.fromarray(im[0])

    def show_mask(self, mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def extract_source_masks(self):
        os.makedirs(self.output_path, exist_ok=True)

        images = list(Path(self.datat_path).glob("*.png")) + list(Path(self.datat_path).glob("*.jpg"))
        images = sorted(images, key=lambda x: int(x.stem))
        video = torch.stack([ToTensor()(Image.open(img).resize(self.resolution)) for img in images]).to("cuda").to(
            dtype=self.precision)

        all_obj_masks = {}
        for obj_name, obj_prompt in self.object_of_interest_dict.items():
            obj_name_for_dir = obj_name.replace(" ", "_")
            obj_mask_dir = os.path.join(self.output_path, obj_name_for_dir)
            os.makedirs(obj_mask_dir, exist_ok=True)
            prompt = obj_prompt
            if self.semantic_level:
                prompt = "[semantic] " + prompt

            # initialize model and tokenizer
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device=self.model.device)

            predictor = self.model.visual_model
            outputs = {}
            for frame_idx in self.id_list:
                inference_state = predictor.init_state(video=video)
                predictor.reset_state(inference_state)
                image_np = (video[frame_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                image_beit = self.beit3_preprocess(image_np).to(dtype=self.model.dtype, device=self.model.device)

                output = self.model.mm_extractor.beit3(visual_tokens=image_beit.unsqueeze(0), textual_tokens=input_ids,
                                                  text_padding_position=torch.zeros_like(input_ids))
                feat = self.model.text_hidden_fcs[0](output["encoder_out"][:, :1, ...])

                _, out_obj_ids, out_mask_logits = predictor.add_new_text(
                    inference_state=inference_state,
                    frame_idx=frame_idx,  # the frame index we interact with
                    obj_id=1,  # give a unique id to each object we interact with (it can be any integers)
                    text=feat
                )

                # run propagation throughout the video and collect the results in a dict
                video_segments = {}  # video_segments contains the per-frame segmentation results
                for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                                start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                outputs[f"{frame_idx}"] = video_segments

                # save visualization

            os.makedirs(os.path.join(obj_mask_dir, "for_masks"), exist_ok=True)
            os.makedirs(os.path.join(obj_mask_dir, "for_masks", "fig"), exist_ok=True)
            os.makedirs(os.path.join(obj_mask_dir, "for_masks", "mask"), exist_ok=True)

            plt.close("all")

            masks_img = []
            for i in list(outputs[f"{self.id_list[0]}"].keys()):
                plt.figure(figsize=(6, 4))
                plt.title(f"frame {i}")
                plt.imshow(Image.open(images[i]))
                out_mask = None
                for j in self.id_list:
                    out_mask = out_mask + outputs[f"{j}"][i][1] if out_mask is not None else outputs[f"{j}"][i][1]
                    masks_img.append(self.return_mask_img(out_mask))

                self.show_mask(out_mask, plt.gca())
                plt.savefig(str(obj_mask_dir + f"/for_masks/fig/{i:05d}.png"))
                torch.save(out_mask, str(obj_mask_dir + f"/for_masks/mask/{i:05d}.pt"))
                self.return_mask_img(out_mask).save(str(obj_mask_dir + f"/for_masks/mask/{i:05d}.png"))
            plt.close("all")
            masks = torch.stack([ToTensor()(m) for m in masks_img]).to("cuda")
            all_obj_masks[obj_name] = masks

        return all_obj_masks


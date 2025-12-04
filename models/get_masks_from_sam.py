import os
os.environ["HF_HUB_OFFLINE"] = "1"  # Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import sys

sys.path.append("third_party/evfsam2")
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
from torchvision.transforms import ToTensor
from PIL import Image


video_codec = "libx264"
video_options = {
    "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
    "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
}


class SAMMaskGenerator:
    def __init__(self, pipeline, precision=torch.float16, image_size=224, obj_prompt="puppy", id_for_sam="all_frames",
                 semantic_level=False):
        self.pipeline = pipeline
        self.precision = precision
        self.image_size = image_size
        self.obj_prompt = "[semantic] " + obj_prompt if semantic_level else obj_prompt
        self.id_for_sam = id_for_sam
        if self.id_for_sam == "all_frames":
            self.id_list = [j for j in range(49)]
        elif self.id_for_sam == "3_key_frames":
            self.id_list = [0, 24, 48]
        elif self.id_for_sam == "5_key_frames":
            self.id_list = [0, 12, 24, 36, 48]
        elif self.id_for_sam == "first_frame":
            self.id_list = [0]

        # frame_count = decoded_video.shape[0]
        # self.id_list = {
        #     "all_frames": list(range(frame_count)),
        #     "3_key_frames": [0, frame_count // 2, frame_count - 1],
        #     "5_key_frames": [0, frame_count // 4, frame_count // 2, 3 * frame_count // 4, frame_count - 1],
        #     "first_frame": [0]
        # }[self.id_for_sam]

        self.tokenizer, self.model = self.init_models(version='YxZhang/evf-sam2-multitask')
        # self.tokenizer, self.model = self.init_models(version=version='YxZhang/evf-sam2')
        self.input_ids = self.tokenizer(self.obj_prompt, return_tensors="pt")["input_ids"].to(device=self.model.device)

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

    def decode_video(self, video_latent):
        with torch.no_grad():
            decoded_video = self.pipeline.decode_latents(video_latent)
            decoded_video = self.pipeline.video_processor.postprocess_video(video=decoded_video,
                                                                            output_type="pt").squeeze(0).to("cuda")
        return decoded_video

    def return_mask_img(self, mask):
        im = (mask * 255.0).round().astype("uint8")
        return Image.fromarray(im[0])

    def generate_masks(self, video_latent):
        decoded_video = self.decode_video(video_latent)
        predictor = self.model.visual_model

        outputs = {}

        for frame_idx in self.id_list:
            inference_state = predictor.init_state(video=decoded_video)
            predictor.reset_state(inference_state)

            image_np = (decoded_video[frame_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            image_beit = self.beit3_preprocess(image_np).to(dtype=self.model.dtype,
                                                                        device=self.model.device)

            feat_output = self.model.mm_extractor.beit3(
                visual_tokens=image_beit.unsqueeze(0),
                textual_tokens=self.input_ids,
                text_padding_position=torch.zeros_like(self.input_ids)
            )
            feat = self.model.text_hidden_fcs[0](feat_output["encoder_out"][:, :1, ...])

            predictor.add_new_text(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=1,
                text=feat
            )

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, start_frame_idx=0):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            outputs[f"{frame_idx}"] = video_segments

        masks_img = []
        for i in list(outputs[f"{self.id_list[0]}"].keys()):
            out_mask = sum(outputs[f"{j}"][i][1] for j in self.id_list)
            masks_img.append(self.return_mask_img(out_mask))

        masks = torch.stack([ToTensor()(m) for m in masks_img]).to("cuda")

        return masks









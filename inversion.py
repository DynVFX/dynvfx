import os
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from torchvision.io import read_video
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler

DEVICE = "cuda:0"


class PreprocessCogVideo():
    def __init__(self, config):
        super().__init__()
        self.device = DEVICE
        if config["precision"] == "bfloat16":
            self.precision = torch.bfloat16
        elif config["precision"] == "float16":
            self.precision = torch.float16
        elif config["precision"] == "float32":
            self.precision = torch.float32
        self.config = config
        self.resolution = (self.config["resolution"][0], self.config["resolution"][1])


        self.model_key = f"THUDM/CogVideoX-5b"
        self.pipeline = CogVideoXPipeline.from_pretrained(self.model_key, torch_dtype=self.precision, local_files_only=True).to(self.device)

        self.scheduler = CogVideoXDDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing", torch_dtype=self.precision)
        self.pipeline.transformer.to(memory_format=torch.channels_last)
        self.pipeline.vae.num_sample_frames_batch_size = self.config["max_number_of_frames"]

        self.vae = self.pipeline.vae

        self.pipeline.scheduler = self.scheduler
        self.pipeline.enable_model_cpu_offload(device=self.device)


    # DDIM Inversion
    @torch.no_grad()
    def init_prompt(self, prompt):
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt,
            "",
            True,
            num_videos_per_prompt=1,
            device=self.device,
        )
        context = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        return context

    def get_v_pred_single(self, latents, t, context):
        height: int = 480
        width: int = 720
        image_rotary_emb = (
            self.pipeline._prepare_rotary_positional_embeddings(height, width, latents.size(1), self.device)
            if self.pipeline.transformer.config.use_rotary_positional_embeddings
            else None
        )
        v_pred = self.pipeline.transformer(
            hidden_states=latents,
            encoder_hidden_states=context,
            timestep=t.expand(latents.shape[0]),
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        return v_pred

    @torch.no_grad()
    def ddim_inversion(self, cond, latent, export_schedule):
        timesteps = reversed(self.scheduler.timesteps)
        latents = []
        save_latents_path = Path(self.config["output_path"])
        os.makedirs(save_latents_path, exist_ok=True)


        with torch.autocast(device_type=self.device, dtype=self.precision):
            for i, t in enumerate(tqdm(timesteps)):
                cond_batch = cond.repeat(latent.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                t_prev = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t_prev = 1 - alpha_prod_t_prev # sigma_prev
                if t_prev >= 0:
                    a_t_prev = ((1 - alpha_prod_t) / (1 - alpha_prod_t_prev)) ** 0.5
                else:
                    a_t_prev = 0.

                b_t_prev = alpha_prod_t ** 0.5 - alpha_prod_t_prev ** 0.5 * a_t_prev

                v_pred = self.get_v_pred_single(latent, t, cond_batch)
                if False in torch.isfinite(v_pred):
                    raise RuntimeError("NaN or Inf in noise_pred")

                pred_x0 = (alpha_prod_t_prev ** 0.5) * latent - (beta_prod_t_prev ** 0.5) * v_pred
                latent = a_t_prev * latent + b_t_prev * pred_x0

                if t in export_schedule:
                    torch.save(latent, os.path.join(save_latents_path, f"latent_{t}.pt"))

                latents.append(latent)
        return latents

    @torch.no_grad()
    def extract_latents(self):
        data_path = config["data_path"]
        if data_path.endswith(".mp4"):
            video = read_video(data_path, pts_unit="sec")[0].permute(0, 3, 1, 2).cuda() / 255
            video = [ToPILImage()(video[i]).resize(self.resolution) for i in range(video.shape[0])]
        else:
            images = list(Path(data_path).glob("*.png")) + list(Path(data_path).glob("*.jpg"))
            images = sorted(images, key=lambda x: int(x.stem))
            video = [Image.open(img).resize(self.resolution) for img in images]

        video = video[:self.config["max_number_of_frames"]]
        video = self.pipeline.video_processor.preprocess_video(video).to(self.device)
        video_latent = self.vae.config.scaling_factor * self.vae.encode(video.to(self.vae.dtype)).latent_dist.sample()
        video_latent = video_latent.permute((0, 2, 1, 3, 4)) # b f c h w
        os.makedirs(self.config["output_path"], exist_ok=True)

        self.scheduler.set_timesteps(self.config["n_save_timesteps"], device=self.device)
        export_schedule = self.scheduler.timesteps.clone()
        self.scheduler.set_timesteps(self.config["n_inversion_timesteps"], device=self.device)

        context = self.init_prompt(self.config["inversion_prompt"])
        uncond_embeddings, cond_embeddings = context.chunk(2)
        _ = self.ddim_inversion(
            cond=cond_embeddings, latent=video_latent, export_schedule=export_schedule,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--default_config_path", type=str, default="configs/inversion_config.yaml")
    parser.add_argument("--user_config_path", type=str, default="configs/user_config.yaml")

    opt = parser.parse_args()
    config = OmegaConf.load(opt.default_config_path)
    user_config = OmegaConf.load(opt.user_config_path)

    config.update(
        {
            "data_path": user_config["data_path"],
            "output_path": user_config["latents_path"],
        }
    )
    model = PreprocessCogVideo(config)
    model.extract_latents()


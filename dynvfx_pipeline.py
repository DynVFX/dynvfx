import argparse
import copy
import os.path
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm

# Third-party imports
import sys
sys.path.append("third_party/evfsam2")
from models.get_masks_from_sam import SAMMaskGenerator
# from get_masks_from_sam import return_masks_from_sam

# Local imports
from utilities.attention_utils import (
    register_attention_guidance,
    get_text_embeds,
    register_module_property,
)
from utilities.masking_utils import (
    return_latent_mask,
    apply_clustering,
    apply_morphological_ops, extract_source_masks,
)
from utilities.utils import (
    load_our_latents_t,
    save_video,
    get_timesteps,
    seed_everything, initialize_precision, load_video_frames,
)


class DynVFXPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]
        self.resolution = (self.config["resolution"][0], self.config["resolution"][1])

        # Set up precision
        self.precision = initialize_precision(config["precision"])

        # Initialize diffusion pipeline
        self.init_diffusion_pipeline()

        # Setup schedulers
        self.init_schedulers()

        # Load data
        self.input_tensor, self.source_latents, self.decoded_source_frames = self.load_data()

        # Load masks
        self.objects_mask_dict, self.masks_for_attn = self.load_latent_mask()

        # Initialize text embeddings
        with torch.no_grad():
            self.target_text_embeds = get_text_embeds(self.pipeline, config["target_prompt"], config["negative_prompt"])
            self.source_text_embeds = get_text_embeds(self.pipeline, config["prompt"], config["negative_prompt"])

        # Initialize SAM mask generator
        self.sam_mask_gen = SAMMaskGenerator(
            self.pipeline,
            self.precision,
            obj_prompt=self.config["target_word"],
            id_for_sam=self.config["id_for_sam"],
            semantic_level=self.config["semantic_level"],
        )

    def init_diffusion_pipeline(self):
        """Initialize the CogVideoX diffusion pipeline."""

        self.pipeline = CogVideoXPipeline.from_pretrained(
            self.config["model_key"],
            torch_dtype=self.precision,
            local_files_only=True,
        ).to(self.device)

        self.pipeline.scheduler = CogVideoXDDIMScheduler.from_config(
            self.pipeline.scheduler.config
        )
        self.pipeline.scheduler.set_timesteps(self.config["n_timesteps"], device=self.device)
        self.pipeline.vae.num_sample_frames_batch_size = self.config["max_frames"]

        # Store references
        self.vae = self.pipeline.vae
        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.transformer = self.pipeline.transformer

        # CPU offload if configured
        if self.config["with_cpu_offload"]:
            self.pipeline.enable_model_cpu_offload(device=self.device)
        # self.pipeline.vae.enable_tiling()
        # print("Model loaded")

    def init_schedulers(self):
        self.sampling_scheduler = copy.deepcopy(self.pipeline.scheduler)
        self.scheduler_full_timesteps = self.pipeline.scheduler.timesteps
        self.scheduler_order = self.pipeline.scheduler.order
        _, self.injection_schedule = get_timesteps(
            self.scheduler_full_timesteps,
            self.scheduler_order,
            self.config["n_timesteps"],
            self.config["max_guidance_timestep"],
            self.config["min_injection_timestep"],
        )
        _, self.masked_injection_schedule = get_timesteps(
            self.scheduler_full_timesteps,
            self.scheduler_order,
            self.config["n_timesteps"],
            self.config["max_masked_injection_timestep"],
            self.config["min_injection_timestep"],
        )
        self.min_inject = None
        self.sampling_scheduler.timesteps, self.guidance_schedule = get_timesteps(
            self.scheduler_full_timesteps,
            self.scheduler_order,
            self.config["n_timesteps"],
            self.config["max_guidance_timestep"],
            self.config["min_guidance_timestep"],
        )
        # self.sample_loop_scheduler = copy.deepcopy(self.pipeline.scheduler)
        self.sample_loop_scheduler = CogVideoXDDIMScheduler.from_config(self.pipeline.scheduler.config)

        self.sdedit_strengths_dict = {}
        for i, t in enumerate(reversed(self.sampling_scheduler.timesteps)):
            self.sdedit_strengths_dict[f"{t}"] = (i + 1) / self.config["n_timesteps"]

    @torch.no_grad()
    def load_latent_mask(self):
        objects_mask_dict = {}
        name_of_mask, masks = extract_source_masks(
            self.config["object_of_interest_mask_dict"],
            self.config["masks_dir"],
            self.config["masks_internal_path"],
            self.config["max_frames"],
            self.config["with_mask_opening"],
            self.config["with_mask_erosion"],
            self.precision
        )
        latent_masks = self.encode(masks.repeat(1, 3, 1, 1))
        latent_masks, low_res_masks = return_latent_mask(latent_masks, masks)
        objects_mask_dict[name_of_mask] = (masks, latent_masks, low_res_masks)
        masks_for_attn = low_res_masks
        return objects_mask_dict, masks_for_attn

    @torch.no_grad()
    def load_data(self):
        # Load video frames
        video = load_video_frames(
            self.config["data_path"],
            self.resolution,
            self.config["max_frames"],
            self.device,
        )

        # Encode to latent space
        source_latents = self.encode(video)

        # Decode for visualization
        decoded_source_frames = self.pipeline.decode_latents(source_latents)
        decoded_source_frames = self.pipeline.video_processor.postprocess_video(
            video=decoded_source_frames, output_type="np"
        )
        decoded_source_frames = (decoded_source_frames[0] * 255).astype(np.uint8)  # f h w c numpy [0,255]
        return video, source_latents, decoded_source_frames

    @torch.no_grad()
    def encode(self, video):
        height, width = self.resolution[1], self.resolution[0]
        preprocessed_video = self.pipeline.video_processor.preprocess_video(video, height=height, width=width).to(
            self.device, self.vae.dtype
        )
        video_latent = self.vae.config.scaling_factor * self.vae.encode(preprocessed_video).latent_dist.sample()
        video_latent = video_latent.permute((0, 2, 1, 3, 4))
        return video_latent

    @torch.no_grad()
    def denoise_step(self,
                     x, t):
        current_scheduler = self.sample_loop_scheduler
        bsz, frames, channel, height, width = x.shape

        # Prepare rotary embeddings
        image_rotary_emb = (
            self.pipeline._prepare_rotary_positional_embeddings(self.resolution[1], self.resolution[0], frames, self.device)
            if self.pipeline.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # Prepare latent input (conditional + source)
        if self.config["inject_from_invert"]:
            inverted_latent = load_our_latents_t(t.item(), self.config["latents_path"]).to(self.precision)
            latent_model_input = torch.cat([x, inverted_latent], dim=0)
        else:
            x_source = self.sampling_scheduler.add_noise(self.source_latents, self.sampling_noise, t)
            latent_model_input = torch.cat([x, x_source], dim=0)

        # Prepare text embeddings
        cond_ebmeds, uncond_embeds = self.target_text_embeds.chunk(2)
        source_cond_ebmeds, _ = self.source_text_embeds.chunk(2)
        text_embeds = torch.cat([cond_ebmeds, source_cond_ebmeds], dim=0)
        uncond_batch = uncond_embeds.repeat(x.shape[0], 1, 1)

        # Conditional forward pass
        v_pred = self.pipeline.transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=text_embeds,
            timestep=t.expand(latent_model_input.shape[0]),
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        v_pred = v_pred.float()
        v_pred_cond, _ = v_pred.chunk(2)

        # Unconditional forward pass
        v_pred_uncond = self.pipeline.transformer(
            hidden_states=x,
            encoder_hidden_states=uncond_batch,
            timestep=t.expand(x.shape[0]),
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
        )[0]
        v_pred_uncond = v_pred_uncond.float()

        # Classifier-free guidance
        v_pred = v_pred_uncond + self.config["guidance_scale"] * (v_pred_cond - v_pred_uncond)

        # Scheduler step
        denoised_latent = current_scheduler.step(v_pred, t, x)["prev_sample"].to(self.precision)
        return denoised_latent

    @torch.no_grad()
    def sampling(self, strength, x_comp, with_inject=False):
        # Setup scheduler
        self.sample_loop_scheduler.set_timesteps(self.config["n_timesteps"], device="cuda")
        self.sample_loop_scheduler.timesteps, self.sample_loop_schedule = get_timesteps(
            self.sample_loop_scheduler.timesteps,
            self.scheduler_order,
            self.config["n_timesteps"],
            max_guidance_timestep=strength,
            min_guidance_timestep=self.config["slim_min_guidance_timestep"],
        )

        # Initialize noise
        x = self.sample_loop_scheduler.add_noise(
            x_comp, self.sampling_noise, self.sample_loop_scheduler.timesteps[0]
        ).to(self.precision)

        # Denoising loop
        for i, t in enumerate(tqdm(self.sample_loop_schedule, desc="Sampling")):
            with torch.autocast(device_type=self.device, dtype=self.precision):
                if with_inject and (t in self.injection_schedule):
                    register_module_property(self.pipeline, property_name="with_injection", property_value=True)

                    if self.config["mask_injection"] and (t in self.masked_injection_schedule):
                        register_module_property(self.pipeline, property_name="masks_for_attn", property_value=self.masks_for_attn)
                        register_module_property(self.pipeline, property_name="mask_injection",
                                                 property_value=True)
                    else:
                        register_module_property(self.pipeline, property_name="mask_injection",
                                                 property_value=False)
                else:
                    register_module_property(self.pipeline, property_name="with_injection", property_value=False)
                x = self.denoise_step(x, t)

        # Disable injection after completion
        register_module_property(self.pipeline, property_name="with_injection", property_value=False)
        return x

    def update_target_comp(self, x_comp, t):
        self.sampling_noise = torch.randn_like(
            self.source_latents, device=self.device, dtype=self.precision
        )

        # Determine injection state
        with_inject = self.config["with_injection"] and (t > self.config["stop_injection_t"])

        # Apply SDEdit
        x_comp_updated = self.sampling(self.sdedit_strengths_dict[f"{t}"], x_comp, with_inject)

        with torch.no_grad():
            bsz, frames, channel, height, width = x_comp_updated.shape
            # target_mask = (
            #     return_masks_from_sam(
            #         self.pipeline,
            #         x_comp_updated,
            #         self.config["target_word"],
            #         self.precision,
            #         id_for_sam=self.config["id_for_sam"],
            #         semantic_level=self.config["semantic_level"],
            #     )
            #     .to(self.precision)
            #     .to(self.device)
            # )
            target_mask = (self.sam_mask_gen.generate_masks(x_comp_updated).to(self.precision).to(self.device))
            target_mask = apply_morphological_ops(target_mask, with_opening=self.config["with_mask_opening"], with_erosion=self.config["with_mask_erosion"], dtype=self.precision)

            # Generate latent mask via clustering
            latent_target_mask = self.encode(target_mask.repeat(1, 3, 1, 1)).to("cuda")
            flattened_latents = rearrange(latent_target_mask, "b f c h w -> b (f h w) c")
            latent_target_mask = apply_clustering(flattened_latents, target_mask, self.precision, self.device, frames, height, width).to(self.device)
        return x_comp_updated, target_mask, latent_target_mask

    def run(self):
        seed_everything(self.config["seed"])
        if self.config["register_guidance"]:
            if self.config["with_injection"]:
                register_attention_guidance(self)

        output_path = Path(self.config["output_path"])

        # Log source data
        if self.config["with_logger"]:
            Path(f"{output_path}/preprocess").mkdir(parents=True, exist_ok=True)
            self.log_source_data()

        # Initialize residuals
        x_res = torch.zeros_like(self.source_latents, device=self.source_latents.device, dtype=self.precision)
        x_orig = self.source_latents.clone().detach().requires_grad_(False).to(self.precision)

        # Main loop
        for i, t in enumerate(tqdm(self.guidance_schedule, desc="Sampling")):
            if t.item() not in self.config["method_steps"]:
                continue

            with torch.autocast(device_type=self.device, dtype=self.precision):
                x_comp = x_orig.detach() + x_res

                with torch.no_grad():
                    x_comp_updated, target_mask, latent_target_mask = self.update_target_comp(x_comp, t)

                # Update residual
                x_diff = x_comp_updated - x_orig
                x_res = latent_target_mask * x_diff

            # Sanity checks
            assert (not x_res.isnan().any()) and (not x_orig.isnan().any())
            assert (not x_res.isinf().any()) and (not x_orig.isinf().any())

            if self.config["with_logger"]:
                self.logger(x_orig, x_res, x_comp_updated, target_mask, latent_target_mask, t)

        # Generate final output
        x_final = x_orig + x_res

        with torch.no_grad():
            decoded_final = self.pipeline.decode_latents(x_final)
            decoded_final = self.pipeline.video_processor.postprocess_video(
                video=decoded_final, output_type="np"
            )
            decoded_final = (decoded_final[0] * 255).astype(np.uint8)
            save_video(decoded_final, os.path.join(self.config["output_path"], f"result.mp4"))

    @torch.no_grad()
    def log_source_data(self):
        save_video(self.decoded_source_frames, os.path.join(self.config["output_path"], "preprocess", f"input.mp4"))
        for obj_name, (masks, latent_masks, low_res_masks) in self.objects_mask_dict.items():
            save_video(
                (latent_masks.repeat(1, 3, 1, 1).float() * 255).permute(0, 2, 3, 1),
                os.path.join(self.config["output_path"], "preprocess", f"latent_masks_{obj_name}.mp4"),
            )
            save_video(
                (masks.repeat(1, 3, 1, 1).float() * 255).permute(0, 2, 3, 1),
                os.path.join(self.config["output_path"], "preprocess", f"masks_{obj_name}.mp4"),
            )

    @torch.no_grad()
    def logger(self, x_orig, x_res, x_comp_updated, target_mask, latent_target_mask, t):
        x_comp = x_orig + x_res
        decoded_x_comp = self.pipeline.decode_latents(x_comp.to(x_res.dtype))
        decoded_x_comp = self.pipeline.video_processor.postprocess_video(video=decoded_x_comp, output_type="np")
        decoded_x_comp = (decoded_x_comp[0] * 255).astype(np.uint8)  # f h w c numpy [0,255]
        save_video(decoded_x_comp, os.path.join(self.config["output_path"], f"decoded_x_comp_t{t}_k{0}.mp4"))

        decoded_x_comp_updated = self.pipeline.decode_latents(x_comp_updated.to(x_res.dtype))
        decoded_x_comp_updated = self.pipeline.video_processor.postprocess_video(
            video=decoded_x_comp_updated, output_type="np"
        )
        decoded_x_comp_updated = (decoded_x_comp_updated[0] * 255).astype(np.uint8)  # f h w c numpy [0,255]
        save_video(
            decoded_x_comp_updated, os.path.join(self.config["output_path"], f"decoded_x_comp_updated_t{t}_k{0}.mp4")
        )

        save_video(
            np.array(255 * (latent_target_mask.float()).repeat(1, 3, 1, 1).detach().cpu()).transpose(0, 2, 3, 1),
            os.path.join(self.config["output_path"], f"latent_target_mask_t{t}_k{0}.mp4"),
        )

        save_video(
            np.array(255 * (target_mask.float()).repeat(1, 3, 1, 1).detach().cpu()).transpose(0, 2, 3, 1),
            os.path.join(self.config["output_path"], f"target_mask_t{t}_k{0}.mp4"),
        )


def run_pipeline(pipeline_config):
    config = pipeline_config

    output_path = Path(config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config, output_path / "run_config.yaml")

    # Set or generate random seed
    if config.get("seed") is None:
        seed = torch.randint(0, 1000000, (1,)).item()
        config["seed"] = seed
        with open(output_path / "seed.txt", "w") as f:
            f.write(str(seed))

    # Set global seed
    seed_everything(config["seed"])

    pipeline = DynVFXPipeline(config)
    pipeline.run()
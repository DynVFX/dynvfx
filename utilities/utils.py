import gc
import os
import random
from typing import Union
import numpy as np
import torch
from torchvision.io import write_video
from typing import List
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor


video_codec = "libx264"
video_options = {
    "crf": "17",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
    "preset": "slow",  # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
}


def load_our_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f"latent_{t}.pt")
    assert os.path.exists(latents_t_path), f"Missing latent at t {t} path {latents_t_path}"
    latents = torch.load(latents_t_path, weights_only=False)
    return latents

def save_video(video, path):
    write_video(
        path,
        video,
        fps=8,
        video_codec=video_codec,
        options=video_options,
    )


def load_video_frames(data_path, resolution, max_frames, device):
    """
    Load video frames from a directory of images.

    Args:
        data_path: Path to directory containing image frames.
        resolution: Target resolution as (width, height).
        max_frames: Maximum number of frames to load.
        device: Device to load tensors to.

    Returns:
        Video tensor of shape (F, C, H, W) with values in [0, 1].
    """
    data_path = Path(data_path)

    # Find all image files
    images = list(data_path.glob("*.png")) + list(data_path.glob("*.jpg"))
    images = sorted(images, key=lambda x: int(x.stem))

    if not images:
        raise FileNotFoundError(f"No image files found in {data_path}")

    video = torch.stack([ToTensor()(Image.open(img).resize(resolution)) for img in images]).to(device)
    video = video[: max_frames]
    return video

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  #":4096:8" #:16:8


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def isinstance_str(x: object, cls_name: Union[str, List[str]]):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """
    if type(cls_name) == str:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ == cls_name:
                return True
    else:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ in cls_name:
                return True
    return False


def get_timesteps(scheduler_num_train_timesteps, scheduler_order, num_inference_steps, max_guidance_timestep, min_guidance_timestep):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * max_guidance_timestep), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    t_end = int(num_inference_steps * min_guidance_timestep)
    timesteps = scheduler_num_train_timesteps[(t_start * scheduler_order) :]
    if t_end > 0:
        guidance_schedule = scheduler_num_train_timesteps[(t_start * scheduler_order) : (-t_end * scheduler_order)]
    else:
        guidance_schedule = scheduler_num_train_timesteps[(t_start * scheduler_order) :]
    return timesteps, guidance_schedule


def initialize_precision(precision_str):
    """Initialize the precision/dtype for computations."""
    precision_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if precision_str not in precision_map:
        raise ValueError(
            f"Unsupported precision: {precision_str}. "
            f"Choose from {list(precision_map.keys())}"
        )

    return precision_map[precision_str]
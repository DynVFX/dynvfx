import pyamg
import torch
from PIL import Image
from einops import rearrange
from kornia.morphology import dilation, erosion, opening
from sklearn.cluster import KMeans
from torchvision.transforms import ToTensor


def return_mask_img(mask):
    im = (mask * 255.0).round().astype("uint8")
    return Image.fromarray(im)

def return_latent_mask(encoded_masks, masks):
    batch, frames, channel, height, width = encoded_masks.shape
    device = encoded_masks.device
    precision = encoded_masks.dtype
    flattened_latents = rearrange(encoded_masks, "b f c h w -> b (f h w) c")
    latent_masks = apply_clustering(flattened_latents, masks, precision, device, frames, height, width)
    feature_h = height // 2
    feature_w = width // 2
    low_res_masks = torch.zeros(frames, 1, feature_h, feature_w).to(device).to(precision)
    h_scale = feature_h / height
    w_scale = feature_w / width
    for i in range(frames):
        indices = latent_masks[i][0].nonzero()
        scaled_indices = indices * torch.tensor([h_scale, w_scale])
        scaled_indices = scaled_indices.round().long()
        scaled_indices[:, 0] = scaled_indices[:, 0].clamp(0, feature_h - 1)
        scaled_indices[:, 1] = scaled_indices[:, 1].clamp(0, feature_w - 1)
        low_res_masks[i, 0, scaled_indices[:, 0], scaled_indices[:, 1]] = 1
    return (
        latent_masks,
        low_res_masks,
    )


def apply_clustering(flattened_latents, masks, precision, device, frames, height, width):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(flattened_latents[0].float().cpu().numpy())
    latents_clustering = rearrange(
        kmeans.predict(flattened_latents[0].float().cpu().numpy()),
        "(f h w) -> f 1 h w",
        f=frames,
        h=height,
        w=width,
    )
    latent_masks = torch.tensor(latents_clustering).to(precision)
    foreground_count = (masks == 1).sum()
    background_count = (masks == 0).sum()
    majority_label = 1 if foreground_count > background_count else 0
    latent_foreground_count = (latent_masks == 1).sum()
    latent_background_count = (latent_masks == 0).sum()
    latent_majority_label = 1 if latent_foreground_count > latent_background_count else 0
    if latent_majority_label != majority_label:
        latent_masks = 1 - latent_masks
    return latent_masks


def apply_morphological_ops(mask, with_opening, with_erosion, dtype, kernel_size=3):
    kernel = torch.ones(kernel_size, kernel_size, device=mask.device)

    if with_opening:
        mask = opening(mask, kernel=kernel).to(dtype)

    if with_erosion:
        mask = erosion(mask, kernel=kernel).to(dtype)

    return mask

def extract_source_masks(object_of_interest_mask_dict, masks_dir, masks_internal_path, max_frames,
                         with_mask_opening, with_mask_erosion,  precision):
    name_of_mask = ""
    masks = None
    for obj_name, obj_path in object_of_interest_mask_dict.items():
        name_of_mask += obj_name
        obj_path = obj_path.replace(" ", "_")
        masks_path = str(masks_dir + f"/{obj_path}/" + masks_internal_path)
        masks_img = [
            return_mask_img(torch.load(f"{masks_path}/{i:05d}.pt", weights_only=False)[0]) for i in range(max_frames)
        ]
        obj_masks = torch.stack([ToTensor()(m) for m in masks_img]).to(precision)
        masks = obj_masks if masks is None else masks + obj_masks
        masks = apply_morphological_ops(masks, with_opening=with_mask_opening,
                                        with_erosion=with_mask_erosion, dtype=precision)
    return name_of_mask, masks
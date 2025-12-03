import os
import torch.nn as nn
from openai import OpenAI
import base64
from mimetypes import guess_type
from .system_prompts import sys_prompt_composited_content, sys_prompt_for_anchor_mask
import json
from pathlib import Path
from datetime import datetime
import re
from dotenv import load_dotenv

from pydantic import BaseModel

class VFXArtistOutput(BaseModel):
    source_scene_caption: str
    vfx_conversation: str
    composited_scene_caption: str

class MasksOutput(BaseModel):
    # reasoning: str
    source_objects_list: list[str]
    target_object: str

class VFXAssistant(nn.Module):
    def __init__(self):
        super().__init__()
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"
        self.keyframe_indices = [0, 24, 48]
        self.device = "cuda"

    @staticmethod
    def image_to_url(image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream"
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{base64_encoded_data}"

    @staticmethod
    def parse_user_content(user_content: str) -> tuple[str, str]:
        """Extract source scene prompt and new content from user_content."""
        parts = user_content.split("\n\n")
        source_scene_prompt = ""
        new_content = ""

        for part in parts:
            if part.startswith("Source Scene Prompt:"):
                source_scene_prompt = part.replace("Source Scene Prompt:", "").strip()
            elif part.startswith("New Content:"):
                new_content = part.replace("New Content:", "").strip()
                source_scene_prompt = None

        return source_scene_prompt, new_content

    def create_structured_output(self, vfx_output: dict, masks_output: dict, video_path: str,
                               input_text_from_user: str, vfx_system_prompt: str, masks_system_prompt: str):
        """Create a structured output dictionary parsing from user_content."""

        # Parse source scene prompt and new content from user_content
        source_scene_prompt_from_user, new_content_from_user = self.parse_user_content(input_text_from_user)

        if source_scene_prompt_from_user is not None:
            prompt = source_scene_prompt_from_user
        else:
            prompt = vfx_output["source_scene_caption"]

        # Create the nested structure
        structured_data = {
            "input_to_VFX_algorithm":
            {
                "prompt": prompt,
                "edit_prompts": [vfx_output["composited_scene_caption"]],  # List of edit prompts
                "target_words": [masks_output["target_object"]],  # List of target objects
                "object_of_interest_mask_dict": {
                    obj: obj for obj in masks_output["source_objects_list"]
                },
                "object_of_interest_prompts_dict": {
                    obj: obj for obj in masks_output["source_objects_list"]
                },
            },
            "input_to_masks_algorithm":
            {
                "object_of_interest_dict": {
                    obj: obj for obj in masks_output["source_objects_list"]
                },
            },
            "video_path": video_path,
            "input_text_from_user": input_text_from_user,
            "output_from_VFX_system":
                {
                    "vfx_output": vfx_output,
                    "masks_output": masks_output,
                },
        }

        return structured_data

    @staticmethod
    def sanitize_filename(text: str, max_length: int = 50) -> str:
        """Sanitize text for use in filename."""
        # Replace invalid filename characters with underscores
        text = re.sub(r'[<>:"/\\|?*]', '_', text)
        # Replace spaces with underscores
        text = text.replace(' ', '_')
        # Truncate to max length while preserving words
        if len(text) > max_length:
            text = text[:max_length].rsplit('_', 1)[0]
        return text

    def save_outputs(self, vfx_output: dict, masks_output: dict, video_path: str, user_content: str, output_path: str = "outputs"):
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create structured data
        structured_data = self.create_structured_output(
            vfx_output,
            masks_output,
            video_path,
            user_content,
            sys_prompt_composited_content,
            sys_prompt_for_anchor_mask,

        )
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save VFX artist output with structured data
        vfx_file = output_path / f"output_for_vfx_protocol.json"
        with open(vfx_file, 'w') as f:
            json.dump(structured_data, f, indent=2)

        return vfx_file, structured_data



    def run_protocol(self,
                     image_path: str,
                     new_content: str,
                     source_scene_prompt=None,
                     composition_scene_prompt=None,
                     output_path: str = "outputs"):

        user_content = (
                f"New Content: {new_content.strip()}"
            )
        if (source_scene_prompt is None) and  (composition_scene_prompt is None):
            response_vfx = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"{sys_prompt_composited_content}"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": self.image_to_url(f"{image_path}/00000.png"), "detail": "high"}},
                            {"type": "image_url", "image_url": {"url": self.image_to_url(f"{image_path}/00024.png"), "detail": "high"}},
                            {"type": "image_url", "image_url": {"url": self.image_to_url(f"{image_path}/00048.png"), "detail": "high"}},
                            {"type": "text", "text": user_content},
                        ],
                    },
                ],
                response_format=VFXArtistOutput
            )
            vfx_artist_response = response_vfx.choices[0].message.parsed.dict()
        else:
            vfx_artist_response = {"source_scene_caption": source_scene_prompt, "composited_scene_caption": composition_scene_prompt}

        masks_content = f"""
        Source scene caption: {vfx_artist_response["source_scene_caption"]}.  Composited scene caption: {vfx_artist_response["composited_scene_caption"]}
        """
        masks_response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{sys_prompt_for_anchor_mask}"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self.image_to_url(f"{image_path}/00000.png"), "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": self.image_to_url(f"{image_path}/00024.png"), "detail": "high"}},
                        {"type": "image_url", "image_url": {"url": self.image_to_url(f"{image_path}/00048.png"), "detail": "high"}},
                        {"type": "text", "text": masks_content},
                    ],
                },
            ],
            response_format=MasksOutput
        )
        masks_response = masks_response.choices[0].message.parsed.dict()

        # Save outputs with structured data
        vfx_file, structured_data = self.save_outputs(
            vfx_artist_response,
            masks_response,
            image_path,
            user_content,
            output_path,
        )
        print(f"Outputs saved to:\nVFX: {vfx_file}")
        return vfx_file, structured_data
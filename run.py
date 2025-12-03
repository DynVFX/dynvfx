import argparse
import copy
import uuid
from pathlib import Path
import json
import torch
from omegaconf import OmegaConf

from dynvfx_pipeline import run_pipeline
from utilities.utils import seed_everything
from vfx_assistant.protocol import VFXAssistant
from models.get_source_mask import SourceMaskExtractor


def run_vfx_assistant(data_path, new_content, output_path, target_folder):
    print("\n Running VFX Assistant...")

    output_path = f"{output_path}/{target_folder}"

    assistant = VFXAssistant()

    protocol_path, protocol = assistant.run_protocol(
        image_path=data_path,
        new_content=new_content,
        output_path=output_path,
    )

    print(f" Protocol generated")
    return protocol, protocol_path


def extract_source_masks(data_path, protocol, masks_output_dir):
    """Extract source masks using EVF-SAM."""
    print("\n Extracting source masks with EVF-SAM...")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        source_mask_extractor = SourceMaskExtractor(
            datat_path=data_path,
            output_path=Path(masks_output_dir),
            precision="bf16",
            object_of_interest_dict=protocol["input_to_masks_algorithm"]["object_of_interest_dict"],
        )

        masks = source_mask_extractor.extract_source_masks()

    print("Masks extracted")
    return masks


def load_protocol(protocol_path):
    """Load protocol from JSON file."""
    with open(Path(protocol_path), 'r') as f:
        protocol = json.load(f)
    return protocol


def prepare_pipeline_config(base_config, user_config, protocol, protocol_path):
    # Prepare config for DynVFX pipeline
    input_to_alg = protocol["input_to_VFX_algorithm"]
    output_dir = user_config["output_path"]
    target_folder = user_config["target_folder"]
    run_output_path = Path(f"{output_dir}/{target_folder}")
    Path(run_output_path).mkdir(parents=True, exist_ok=True)

    config_updates = {
        "data_path": user_config["data_path"],
        "new_content": user_config["new_content"],
        "output_path": run_output_path,
        "masks_dir": user_config["masks_dir"],
        "latents_path": user_config["latents_path"],
        "object_of_interest_mask_dict": input_to_alg["object_of_interest_mask_dict"],
        "object_of_interest_prompts_dict": input_to_alg["object_of_interest_prompts_dict"],
        "prompt": input_to_alg["prompt"],
        "target_prompt": input_to_alg["edit_prompts"][0],
        "target_word": input_to_alg["target_words"][0],
        "protocol_path": protocol_path,
    }

    run_config = copy.deepcopy(base_config)
    run_config.update(config_updates)

    OmegaConf.save(run_config, run_output_path / "run_config.yaml")

    return run_config

# Mode: Fully Automated

def mode_auto(user_config, base_config):
    # Fully automated mode - runs entire pipeline in one go.
    print(" DynVFX Fully Automated Mode")

    # Get user inputs from config
    data_path = user_config["data_path"]
    new_content = user_config["new_content"]
    output_path = user_config["output_path"]
    masks_dir = user_config["masks_dir"]
    target_folder = user_config["target_folder"]

    # Step 1: Run VFX Assistant
    print("\n[1/3] VFX Assistant + Text-based Segmentation")
    protocol, protocol_path = run_vfx_assistant(
        data_path=data_path,
        new_content=new_content,
        output_path=output_path,
        target_folder=target_folder
    )

    # Step 2: Extract source masks
    print("\n[2/3] Extracting Masks")
    masks = extract_source_masks(
        data_path=data_path,
        protocol=protocol,
        masks_output_dir=masks_dir
    )

    # Step 3: Prepare and run pipeline
    print("\n[3/3] Running DynVFX Pipeline")
    pipeline_config = prepare_pipeline_config(
        base_config=base_config,
        user_config=user_config,
        protocol=protocol,
        protocol_path=protocol_path
    )

    run_pipeline(pipeline_config)


# Mode: Two-Stage - Generate

def mode_generate(user_config):
    # Stage 1: Generate protocol and extract masks
    print(" Stage 1: Generate & Review Protocol")

    # Get user inputs from config
    data_path = user_config["data_path"]
    new_content = user_config["new_content"]
    output_path = user_config["output_path"]
    masks_dir = user_config["masks_dir"]
    target_folder = user_config["target_folder"]


    # Generate protocol with VFX Assistant
    protocol, protocol_path = run_vfx_assistant(
        data_path=data_path,
        new_content=new_content,
        output_path=output_path,
        target_folder=target_folder
    )

    # Extract source masks
    masks = extract_source_masks(
        data_path=data_path,
        protocol=protocol,
        masks_output_dir=masks_dir
    )

    print(f"Protocol saved to: {protocol_path}")
    print("\nWhen ready, run Stage 2 with the same config:")
    print(f"python run.py execute --user_config_path <your_config.yaml>")


# Mode: Two-Stage - Execute

def mode_execute(user_config, base_config):
    # Stage 2: Execute pipeline with approved protocol.
    print(" Stage 2: Execute with Protocol")

    # Get paths from config
    output_path = user_config["output_path"]
    target_folder = user_config["target_folder"]
    protocol_path = f"{output_path}/{target_folder}/output_for_vfx_protocol.json"

    # Check if protocol exists
    if not Path(protocol_path).exists():
        raise FileNotFoundError(
            f"Protocol not found at {protocol_path}\n"
            f"Please run Stage 1 first: python run.py generate --config_path <your_config.yaml>"
        )

    # Load protocol
    print(f"\n Loading protocol from: {protocol_path}")
    protocol = load_protocol(protocol_path)

    # Prepare and run pipeline
    print("\n Running DynVFX pipeline...")

    pipeline_config = prepare_pipeline_config(
        base_config=base_config,
        user_config=user_config,
        protocol=protocol,
        protocol_path=protocol_path
    )

    run_pipeline(pipeline_config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user_config_path",
        type=str,
        required=True,
        help="Path to user configuration YAML file (user_config.yaml)"
    )
    parser.add_argument(
        "--base_config_path",
        type=str,
        default="configs/base_config.yaml",
        help="Path to base configuration YAML file (base_config.yaml)"
    )

    args = parser.parse_args()

    # Load user configuration
    user_config = OmegaConf.load(args.user_config_path)

    # Load base configuration
    base_config = OmegaConf.load(args.base_config_path)

    mode = user_config["mode"]

    output_path = Path(user_config["output_path"])
    output_path.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(user_config, output_path / "user_config.yaml")
    OmegaConf.save(base_config, output_path / "base_config.yaml")

    # Execute appropriate mode
    if mode == "auto":
        mode_auto(user_config, base_config)
    elif mode == "generate":
        mode_generate(user_config)
    elif mode == "execute":
        mode_execute(user_config, base_config)


if __name__ == "__main__":
    main()
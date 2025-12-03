<div align="center">

# DynVFX: Augmenting Real Videos with Dynamic Content
[![Paper](https://img.shields.io/badge/📄%20Paper-arXiv-red)](https://arxiv.org/abs/2502.03621)
[![Project Page](https://img.shields.io/badge/🌐%20Project-Page-green)](https://dynvfx.github.io/)

[Danah Yatim*](https://www.linkedin.com/in/danah-yatim-4b15231b5/),
[Rafail Fridman*](https://www.linkedin.com/in/rafail-fridman/),
[Omer Bar-Tal](https://omerbt.github.io/),
[Tali Dekel](https://www.weizmann.ac.il/math/dekel/)
<br/>
(*equal contribution)

< ADD VIDEO HERE>
</div>

---

## Overview

This repository contains the official implementation of the paper **DynVFX: Augmenting Real Videos with Dynamic Content**.
>DynVFX augments real-world videos with new dynamic content described by a simple user-provided text instruction. The framework automatically infers where the synthesized content should appear, how it should move, and how it should harmonize at the pixel
level with the scene, without requiring any additional user input. The key idea is to selectively extend the attention mechanism in a pre-trained text-to-video
diffusion model, enforcing the generation to be content-aware of existing scene elements (anchors) from the original video. This allows the model to generate
content that naturally interacts with the environment, producing complex and realistic video edits in a fully automated way.

---

## 📋📥 Setup

### 1. Create New Conda Environment

```
conda create -n dynvfx python=3.12
conda activate dynvfx
```

### 2. Install PyTorch with CUDA support:

```
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```


### 3. Clone the repository and install requirements  

```
#  Clone the repository
git clone https://github.com/DynVFX/dynvfx.git
cd dynvfx

# Install dependencies
pip install -r requirements.txt
```

### 4. API Key
This repository requires an OpenAI API key. If you don't have one, create an OpenAI account and follow the instructions to obtain a key.

Once you have the key, save it in the vfx_assistant/.env file:


```
OPENAI_API_KEY=<your_key>
```
---

# Usage 

---

## ⚙️Configuration
Fill in the following arguments in ``configs/user_config.yaml``:
- ``video_path``: "data/beach_scene/original"          # Path to input video frames directory
- ``new_content``: "a playful golden retriever"        # Text describing content to add
- ``latents_path``: "latents/beach_scene"              # Path to extracted latents (from Section 2)
- ``output_path``: "outputs/beach_dog"                 # Where to save results

See Tips section for configuration options. 

---

## 1. 📹 Input Format (Prepare Your Data)

Your input video should be provided as individual frames in a directory:
```
data/input_frames/
├── 00000.png
├── 00001.png
├── 00002.png
...
├── 00048.png
```

Frames should be named sequentially with zero-padding.

Extract frames from your video:
```
ffmpeg -i input.mp4 -vf fps=8 data/my_video/original/%05d.png
```
---

## 2. 🎞️ Reference Keys and Values Extraction 

To extract the reference keys and values, we first obtain the intermediate latents by inverting the input video:
```bash
python inversion.py --config_path configs/inversion_config.yaml
```

**⚙️ Configuration:** Make sure `video_path` and `latents_path` are set in your `user_config.yaml` file.

**Note:** 
- 🔬 For paper comparison: this step is REQUIRED.
- For general use:
    - 🎯 Run inversion: Best scene alignment and quality.
    - 🎲 Skip inversion: Faster but worse alignment. Could work, could drift.
    - 🧪 Just testing stuff? → You can skip, but it's a gamble. Results could be 🔥, could be 💀.

---

## 3. Run DynVFX (`run.py`)
Given the instructions (→ `new_content`) and input video (→ `video_path`), the automation includes:
1. 🤖 **VLM as a VFX Assistant**
   - 🧠 Interprets the interaction and reasons about integration.
   - 📝 Captions the original scene (→ `prompt`) and target scene (→ `target_prompt`)
   - ⭐ Identifies prominent elements from the original scene and the new content to be added.
2. 🎯 **Text-based Segmentation**
   - Using EVF-SAM obtain source masks of identified elements for AnchorExtAttn.
3. ⚙️ **Pipeline Configuration**
   - Automatically sets up the editing pipeline parameters
4. 🎬 **Run the Edit Pipeline**
   - Runs iterative refinement with AnchorExtAttn
   - Outputs the final augmented video

### Option 1: Fully Automated Mode

For users who want everything automated:

```
python run.py auto --config_path user_config.yaml
```

### Option 2: Preview & Execute Mode

For users who want to first preview 🤖 **VLM as VFX Assistant + Text-based Segmentation** outputs, then execute with protocol:

**Stage 1: Run protocol and extract masks**
```
python run.py generate --config_path user_config.yaml
```

👀 ***Check out the protocol outputs***

**Stage 2: Execute with protocol**  
```
python run.py execute --config_path user_config.yaml
```
---

## Tips

---

## 📊 Results & Visualization

You can optionally save intermediate visualizations. By updating ``with_logger: True`` in ``configs/example_config``. 

In ``output_path`` the following will be saved:
- Input video, source masks and latent source masks.
- Intermediate sample, target mask, target latent mask and result. 

---
## 📁 Project Structure

```
dynvfx/
├── __init__.py              # Package entry point
├── run.py                   # Config-based pipeline CLI
├── run_auto.py              # 🆕 Fully automated CLI
├── run_interactive.py       # 🆕 Two-stage interactive CLI
├── generate_protocol.py     # VFX Assistant CLI
├── core/
│   ├── constants.py         # Constants, enums, and type definitions
│   └── pipeline.py          # Main DynVFXPipeline class
├── configs/
│   └── config.py            # Configuration dataclasses with validation
├── models/
│   ├── attention.py         # Extended attention mechanisms
│   └── sam_generator.py     # SAM-based mask generation
├── assistant/               # VFX Assistant module
│   ├── vfx_assistant.py     # Main VFXAssistant class
│   ├── prompts.py           # System prompts for GPT
│   └── models.py            # Pydantic response models
└── utils/
    ├── video_io.py          # Video loading/saving utilities
    ├── mask_processing.py   # Mask generation and processing
    └── tracking.py          # Experiment tracking utilities
```
---

## 🙏 Credits

This work is based on:
- [EVF-SAM](https://github.com/hustvl/EVF-SAM) - Base text-prompted segmentation model
- [CogVideoX-5B](https://github.com/zai-org/CogVideo) - Base text-to-video model
- [ChatGPT](add here) - VLM model

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{yatim2025dynvfxaugmentingrealvideos,
      title={DynVFX: Augmenting Real Videos with Dynamic Content}, 
      author={Danah Yatim and Rafail Fridman and Omer Bar-Tal and Tali Dekel},
      year={2025},
      eprint={2502.03621},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.03621}, 
}
```

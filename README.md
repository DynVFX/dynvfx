<div align="center">

# DynVFX: Augmenting Real Videos with Dynamic Content
### ✨ SIGGRAPH Asia 2025 ✨ 
#### [Danah Yatim*](https://www.linkedin.com/in/danah-yatim-4b15231b5/), [Rafail Fridman*](https://www.linkedin.com/in/rafail-fridman/), [Omer Bar-Tal](https://omerbt.github.io/), [Tali Dekel](https://www.weizmann.ac.il/math/dekel/) <br/> Weizmann Institute of Science <br/> (\* equal contribution)

[![Website](https://img.shields.io/badge/🌐%20Web-dynvfx.github.io-blue)](https://dynvfx.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.03621-b31b1b.svg)](https://arxiv.org/abs/2502.03621)
[![Conference](https://img.shields.io/badge/🎉%20SA-2025-ff69b4)](https://asia.siggraph.org/2025/)
[![Supp Videos](https://img.shields.io/badge/🎥%20Supp-Videos-purple)](https://dynvfx.github.io/sm/index.html)
![Pytorch](https://img.shields.io/badge/PyTorch->=2.4.0-Red?logo=pytorch)

[//]: # ([![Supp PDF]&#40;https://img.shields.io/badge/📄%20Supp-PDF-orange&#41;]&#40;https://dynvfx.github.io/sm/index.html&#41;)

https://github.com/user-attachments/assets/ab6b34fc-fff1-46d1-97c5-9b393325b3f5 
</div>



This repository contains the official implementation of the paper **DynVFX: Augmenting Real Videos with Dynamic Content**

>**DynVFX** augments real-world videos with new dynamic content described by a simple user-provided text instruction. The framework automatically infers where the synthesized content should appear, how it should move, and how it should harmonize at the pixel
level with the scene, without requiring any additional user input. The key idea is to selectively extend the attention mechanism in a pre-trained text-to-video
diffusion model, enforcing the generation to be content-aware of existing scene elements (anchors) from the original video. This allows the model to generate
content that naturally interacts with the environment, producing complex and realistic video edits in a fully automated way.

For more, visit the [project webpage](https://dynvfx.github.io/).

<div align="center">
    
https://github.com/user-attachments/assets/35f4e598-dd2d-40f6-a8a1-76cf99f7e20a

</div>

---

## Setup 🔧 

### Create New Conda Environment

```
conda create -n dynvfx python=3.12
conda activate dynvfx
```

### Install PyTorch with CUDA:

```
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```


### Clone Repository and Install Dependencies 

```
#  Clone the repository
git clone https://github.com/DynVFX/dynvfx.git
cd dynvfx

# Install dependencies
pip install -r requirements.txt
```
### Install EVF-SAM2

```
# Clone EVF-SAM into third_party directory
git clone https://github.com/hustvl/EVF-SAM third_party/evfsam2
cd third_party/evfsam2

# Install EVF-SAM dependencies
pip install -e .

# Return to project root
cd ../..
```

The model weights (`YxZhang/evf-sam2-multitask`) will be downloaded automatically on first run.

### Set Up OpenAI API Keyy
This repository uses OpenAI's GPT-4o as the VFX Assistant. Create an API key at [OpenAI Platform](https://platform.openai.com/settings/organization/api-keys).

Save your key in `vfx_assistant/.env`:

```
OPENAI_API_KEY=<your_key>
```
---

## Quick Start 🚀

```
# 1. Prepare your video frames (720x480, 49 frames at 8fps)
ffmpeg -i input.mp4 -vf "scale=720:480,fps=8" data/my_video/%05d.png

# 2. Edit configs/user_config.yaml with your paths and desired content

# 3. Run inversion to extract refference keys and values
python inversion.py --user_config_path configs/user_config.yaml

# 4. Run DynVFX
python run.py --user_config_path configs/user_config.yaml
```
---

# Usage 

---

## Configuration ⚙️
Edit `configs/user_config.yaml` with the following parameters:

| Parameter | Description                                              |
|-----------|----------------------------------------------------------|
| `data_path` | Path to input video frames directory                     |
| `new_content` | Text instruction describing new content to add           |
| `output_path` | Directory where output files will be saved               |
| `target_folder` | name of edit, file name where edited video will be saved |
| `masks_dir` | Directory for prominent elements segmentation masks      |
| `latents_path` | Directory for inverted latents                           |
| `mode` | Run mode: `"auto"`, `"generate"`, or `"execute"`         |

See Tips section for configuration options. 

---

## Input Format

Your input video should be provided as individual frames in a directory:
```
data/input_frames/
├── 00000.png
...
├── 00048.png
```

The method works best with:
- **Resolution**: 720×480
- **Frame rate**: 8 fps  
- **Frame count**: 49 frames (~6 seconds)

Resize the video and extract the frames:
```
ffmpeg -i input.mp4 -vf "scale=720:480,fps=8" data/my_video/original/%05d.png
```
---

## Reference Keys and Values Extraction 

To extract the reference keys and values, we first obtain the intermediate latents by inverting the input video:
```
python inversion.py --user_config_path configs/user_config.yaml
```

**Configuration** - Make sure `video_path` and `latents_path` are set in your `user_config.yaml` file.

> **Note:**
> - 🔬 **For paper comparison**: This step is REQUIRED
> - 🎯 **For best quality**: Run inversion for optimal scene alignment
> - 🎲 **For quick testing**: Can be skipped, but results may drift

---

## 3. Running DynVFX 🎬
The pipeline consists of three stages:
1. **🤖 VFX Assistant** — GPT-4o interprets the edit instruction and generates captions
2. **🎭 Text-based Segmentation** — EVF-SAM extracts masks of scene elements
3. **🎬 DynVFX Pipeline** — Iterative refinement with AnchorExtAttn 

### Option A: Fully Automated Mode

Run the entire pipeline in one command:
```yaml
# In configs/user_config.yaml
mode: "auto"
```
```
python run.py --user_config_path configs/user_config.yaml
```

### Option B: Preview & Execute Mode

**Stage 1: Generate 🤖 **VFX Assistant + EVF-SAM** outputs and review

```yaml
# In configs/user_config.yaml
mode: "generate"
```

```
python run.py --user_config_path configs/user_config.yaml
```

👀 Review the generated protocol at `output_path/output_for_vfx_protocol.json` and masks in `masks_dir`.


**Stage 2: Execute with the approved protocol**

```yaml
# In configs/user_config.yaml
mode: "execute"
```

```
python run.py --user_config_path configs/user_config.yaml
```
---

## Repository Structure 📁

```
dynvfx/
├── configs/
│   ├── base_config.yaml      # Pipeline hyperparameters
│   ├── user_config.yaml      # User-specific settings
│   └── inversion_config.yaml # Inversion settings
├── models/
│   ├── get_masks_from_sam.py # SAM mask generation
│   └── get_source_mask.py    # Source mask extraction
├── utilities/
│   ├── attention_utils.py    # Extended attention modules
│   ├── masking_utils.py      # Mask processing utilities
│   └── utils.py              # General utilities
├── vfx_assistant/
│   ├── protocol.py           # VFX Assistant (GPT-4o)
│   ├── system_prompts.py     # System prompts
│   └── .env                  # API keys (create this)
├── third_party/
│   └── evfsam2/              # EVF-SAM installation
├── dynvfx_pipeline.py        # Main pipeline
├── inversion.py              # DDIM inversion
├── run.py                    # Entry point
└── requirements.txt          # Dependencies
```

---

## Tips



### Intermediate Visualization 📊

Enable logging to save intermediate results:

```yaml
# In configs/base_config.yaml
with_logger: True
```

This saves to `output_path`:
- Input video and source masks
- Intermediate samples and target masks
- Latent mask visualizations

---

## Credits

This work builds on:
- [EVF-SAM](https://github.com/hustvl/EVF-SAM) - Base text-prompted segmentation model
- [CogVideoX-5B](https://github.com/zai-org/CogVideo) - Base text-to-video model
- [ChatGPT](https://platform.openai.com/) - Base visual language model

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

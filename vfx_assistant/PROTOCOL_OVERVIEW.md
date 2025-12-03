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

## 🤖 VFX Assistant

The VFX Assistant uses GPT-4 Vision to automatically generate prompts and configurations.

### Quick Start

```python
from dynvfx.assistant import VFXAssistant

# Initialize assistant
assistant = VFXAssistant()  # Uses OPENAI_API_KEY env variable

# Generate protocol
protocol = assistant.generate_protocol(
    video_path="data/beach_scene/original",
    new_content="a playful golden retriever puppy",
    output_dir="protocols/",
)

# Access generated prompts
print(protocol["output_from_VFX_system"]["vfx_output"]["source_scene_caption"])
print(protocol["output_from_VFX_system"]["vfx_output"]["composited_scene_caption"])
```

### CLI Usage

```bash
# Generate protocol for a video
python generate_protocol.py --video data/beach_scene/original --content "a playful puppy"

# Save to specific directory
python generate_protocol.py --video data/beach_scene/original --content "a playful puppy" --output protocols/

# With custom target name
python generate_protocol.py --video data/beach_scene/original --content "a playful puppy" --name "beach_puppy"

# Generate baseline protocol (for comparison methods)
python generate_protocol.py --video data/beach_scene/original --content "a playful puppy" --baseline

# With pre-defined prompts (skip GPT analysis)
python generate_protocol.py --video data/beach_scene/original \
    --content "a playful puppy" \
    --source-prompt "A woman walking on a beach..." \
    --composition-prompt "A woman walking on a beach with a playful puppy..."
```

### What the Assistant Generates

1. **Source Scene Caption**: Detailed description of the original video
2. **VFX Conversation**: Simulated discussion about how to integrate new content
3. **Composited Scene Caption**: Description of the final edited scene
4. **Foreground Objects**: List of objects for masking (people, animals, etc.)
5. **Target Object**: The new content being added

### Output Format

The assistant generates a JSON protocol file that can be used directly with DynVFX:

```json
{
  "input_to_VFX_algorithm": {
    "data/beach_scene/original": {
      "prompt": "A woman in a light floral dress...",
      "edit_prompts": ["A woman walking with a playful puppy..."],
      "target_words": ["playful golden retriever puppy"],
      "object_of_interest_mask_dict": {
        "woman": "woman",
        "water": "water"
      }
    }
  },
  "output_from_VFX_system": {
    "vfx_output": {
      "source_scene_caption": "...",
      "vfx_conversation": "...",
      "composited_scene_caption": "..."
    }
  }
}
```
---

## Tips

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

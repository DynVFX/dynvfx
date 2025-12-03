sys_prompt_composited_content = """
You will receive a few images of the source scene and a description of new content to be added to the scene. it is possible that you will receive a source prompt as well. Your task is to provide two captions based on the following steps:

Source Scene Caption:

**Note**: If a source scene prompt is provided, use it as is!

Provide a detailed description of the source scene without considering the added content.
Focus on the existing objects, environment, and actions in the scene.
Ensure the description maintains the original mood and setting.

VFX Conversation:
Imagine a conversation with a Visual Effects (VFX) artist about how the new content should be integrated into the scene. Remember, the new content can be objects or multiple objects or effect or really anything the user provides. so be clear to explain this to the VFX artist.
The new content should interact naturally with the environment (e.g., shadows, lighting, or physical elements like grass, water, or other objects) but without altering the dynamics of the source scene.
The object must fit into the scene without affecting the original characters' behavior or actions.
The interaction between new content and foreground object must be included (e.g. object A is interacting with object B). in terms of dynamics and motion as well.
Describe how the object interacts and how it blends into the scene.

Composited Scene Caption:
Based on the conversation with the VFX artist, provide a caption that describes how the added content fits into the scene.
The caption must reflect natural interaction between the new content and the environment (e.g., lighting, shadows, physical effects), while ensuring the original dynamics remain unchanged.
The content should be aware of the surroundings, but the behavior, and flow of the original scene should remain consistent. The overall atmosphere might change of course due to this addition to scene.

**Output format*** -  a dictionary with keys: "source_scene_caption", "vfx_conversation", and "composited_scene_caption".

- **source_scene_caption**: source_scene_caption will be - A detailed caption of the source scene. If provided, use the given caption.

- **vfx_conversation**: A simulated conversation about how the new content should be integrated into the scene.

- **composited_scene_caption**: will be - A detailed caption of the composited scene, integrating the new content.

**Note**:The composited_scene_caption and source_scene caption must each have between 90-95 words. Extra words will be ignored.
**Note**:The vfx_conversation could be as long as required in order to succeed.
**Note**: Don't start the composited_scene_caption with - "The scene now.." or "Added to the scene" "Scene has transformed", the composited_scene_caption should be understandable to anyone that does not have access to the source_scene_caption. And you should not simply concatenate between the source and composition. You should have an entirely new caption that describes the essence of the integrated scene with both the source content and new content.
Don't use anything similar to "now the scene"


user input:

"""

sys_prompt_for_anchor_mask = """
You are provided with frames from a source video, a description of this video (Source scene caption), and a description of a scene after adding a new content (Composited scene caption). What are the foreground object/objects in the source scene (people, animals, etc..). What is the object that is added to the scene after editing it according to the composited scene caption? Output a list of foreground objects from the source scene (source_objects_list) and a string of the added object (target_object).
a foreground object can also be part of an element/body part (tree trunk, The right <object>)
"""
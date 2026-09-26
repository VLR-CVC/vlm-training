import re

from megatron.energon import Cooker, basic_sample_keys, stateless
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample

import torch

@edataclass
class EnergonSample(Sample):
    image: torch.Tensor | None
    messages: list
    images: list | None = None

@stateless
def cooker_llava_recap(sample: dict) -> EnergonSample:

    caption = sample['json']['conversations'][1]['value']
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": caption}
        ]},
    ]

    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@stateless
def cooker_llava_imagenet(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"} 
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": sample['txt']}
        ]},
    ]
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@stateless
def cooker_onevision_instruct(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}

    has_image = sample.get('jpg') is not None

    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    image_added = False

    for turn in sample['json']['conversations']:
        raw_role = turn.get('from') or turn.get('role') or 'user'
        role = role_map.get(str(raw_role).lower(), 'user')

        text_val = turn.get('value') or turn.get('content') or ''

        wants_image = has_image and not image_added and (
            "<image>" in text_val or role == 'user'
        )
        text_val = text_val.replace("<image>", "").strip()

        content = []

        if wants_image:
            content.append({"type": "image"})
            image_added = True

        if text_val:
            content.append({"type": "text", "text": text_val})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": role, "content": content})

    return EnergonSample(
        **basic_sample_keys(sample),
        image=sample['jpg'] if has_image else None,
        messages=messages,
    )


@stateless
def cooker_captioning(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}
    
    messages = []
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image_added = False
    
    for turn in sample['json']['conversations']:
        raw_role = turn.get('from') or turn.get('role') or 'user'
        role = role_map.get(str(raw_role).lower(), 'user')
        
        text_val = turn.get('value') or turn.get('content') or ''
        
        content = []
        
        if "<image>" in text_val or (role == 'user' and not image_added):
            content.append({"type": "image"})
            text_val = text_val.replace("<image>", "").strip()
            image_added = True
            
        if text_val:
            content.append({"type": "text", "text": text_val})
            
        if not content:
            content.append({"type": "text", "text": ""})
            
        messages.append({"role": role, "content": content})
    
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@stateless
def cooker_ovevision_midtraining(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"} 
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": sample['json']['caption']}
        ]},
    ]
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@stateless
def cooker_olmo_ocr(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"} 
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": sample['txt']}
        ]},
    ]
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )
@stateless
def cooker_finevision(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}

    has_image = sample.get('jpg') is not None

    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    image_added = False

    for turn in sample['json']['conversations']:
        # `or`, not a .get default: some parquet-derived rows carry both schemas with the unused
        # keys present but null ("from": null next to "role": "assistant"), which a default
        # would not fall back from -- every turn became 'user' and the sample had no labels.
        raw_role = turn.get('from') or turn.get('role') or 'user'
        role = role_map.get(str(raw_role).lower(), 'user')

        text_val = turn.get('value') or turn.get('content') or ''

        wants_image = has_image and not image_added and (
            "<image>" in text_val or role == 'user'
        )
        text_val = text_val.replace("<image>", "").strip()

        content = []

        if wants_image:
            content.append({"type": "image"})
            image_added = True

        if text_val:
            content.append({"type": "text", "text": text_val})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": role, "content": content})

    return EnergonSample(
        **basic_sample_keys(sample),
        image=sample['jpg'] if has_image else None,
        messages=messages,
    )

@stateless
def cooker_idl(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"} 
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": sample['txt']}
        ]},
    ]
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

_NEMOTRON_IMG_KEY = re.compile(r"^(\d+)\.\w+$")


@stateless
def cooker_nemotron(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """Nemotron-VLM-Dataset v1/v2/v3, repacked by `utils/prepare_nemotron_energon.py`"""
    keys = sorted((k for k in sample if _NEMOTRON_IMG_KEY.match(k)),
                  key=lambda k: int(k.split(".", 1)[0]))
    images = [sample[k] for k in keys]
    return EnergonSample(
        **basic_sample_keys(sample),
        image=images[0] if len(images) == 1 else None,
        images=images if len(images) > 1 else None,
        messages=nemotron_messages(sample["json"]["messages"], add_system_prompt),
    )

def nemotron_messages(turns: list, add_system_prompt: bool = True) -> list[dict]:
    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    for turn in turns:
        content = []
        for part in turn["content"]:
            # v3 writes text parts as bare strings, v2 as {"type": "text", ...}
            if isinstance(part, str):
                if part:
                    content.append({"type": "text", "text": part})
            elif part.get("type") == "image":
                content.append({"type": "image"})
            elif part.get("text"):
                content.append({"type": "text", "text": part["text"]})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": turn["role"], "content": content})

    return messages


@stateless
def cooker_text(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """Plain-text webdataset, for `models/qwen3_5_text`"""
    text = sample["txt"] if "txt" in sample else sample["json"]["text"]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": ""}]},
        {"role": "assistant", "content": [{"type": "text", "text": text}]},
    ]
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    return EnergonSample(**basic_sample_keys(sample), image=None, messages=messages)


COOKERS = [
    # subflavors can be used to distinguish datasets when using a Metadataset
    Cooker(cooker_text, has_subflavors={"type_dataset": "text"}),
    Cooker(cooker_captioning, has_subflavors={"type_dataset": "synth_cap"}),
    Cooker(cooker_captioning, has_subflavors={"type_dataset": "synth_finevision"}),
    Cooker(cooker_llava_recap, has_subflavors={"type_dataset": "llava_recap"}),
    Cooker(cooker_llava_imagenet, has_subflavors={"type_dataset": "llava_recap_mn5"}),
    Cooker(cooker_onevision_instruct, has_subflavors={"type_dataset": "onevision_instruct"}),
    Cooker(cooker_ovevision_midtraining, has_subflavors={"type_dataset": "onevision_midtraining"}),
    Cooker(cooker_olmo_ocr, has_subflavors={"type_dataset": "olmo_ocr"}),
    Cooker(cooker_finevision, has_subflavors={"type_dataset": "finevision"}),
    Cooker(cooker_idl, has_subflavors={"type_dataset": "idl_ocr"}),
    Cooker(cooker_nemotron, has_subflavors={"type_dataset": "plotqa_cot"}),
    Cooker(cooker_nemotron, has_subflavors={"type_dataset": "nemotron"}),
    Cooker(cooker_nemotron, has_subflavors={"type_dataset": "nemotron_v1"}),
]

from megatron.energon import Cooker, basic_sample_keys, stateless
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample

import torch

@edataclass
class EnergonSample(Sample):
    image: torch.Tensor | None
    messages: list

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
        raw_role = turn.get('from', turn.get('role', 'user'))
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
        raw_role = turn.get('from', turn.get('role', 'user'))
        role = role_map.get(str(raw_role).lower(), 'user')
        
        text_val = turn.get('value', turn.get('content', ''))
        
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
        raw_role = turn.get('from', turn.get('role', 'user'))
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

@stateless
def cooker_nemotron(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """Nemotron-VLM-Dataset v2/v3, repacked by `utils/prepare_nemotron_energon.py`.

    Unlike the other subsets here, the published `messages` are already
    role/content-list shaped, so this is a filter rather than a rebuild: the
    image part carries a filename and a metadata dict that the processor must
    not see. The packer guarantees exactly one image per sample, so a bare
    `{"type": "image"}` resolves unambiguously to `sample['png']`.

    Images stay PNG (the source is PNG; re-encoding to match the `jpg` key the
    other cookers use would be a lossy no-op), hence `sample['png']`.

    Assistant turns keep their `<think>...</think>` spans verbatim -- these are
    CoT subsets and the reasoning is the training signal.
    """
    return EnergonSample(
        **basic_sample_keys(sample),
        image=sample["png"],
        messages=nemotron_messages(sample["json"]["messages"], add_system_prompt),
    )

def nemotron_messages(turns: list, add_system_prompt: bool = True) -> list[dict]:
    """The message rebuild half of `cooker_nemotron`, split out so tooling that
    reads the shards directly (`utils/tokenizer_compare.py`) counts exactly the
    tokens training sees rather than a re-derivation that can drift from it."""
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


COOKERS = [
    # subflavors can be used to distinguish datasets when using a Metadataset
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
]

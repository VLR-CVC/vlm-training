import json
import re

from megatron.energon import Cooker, basic_sample_keys, stateless
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample

import webdataset as _wds
from pathlib import Path as _Path

import torch

_NEMOTRON_IMG_KEY = re.compile(r"^(\d+)\.\w+$")

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
    """FineVision as staged in `FineVision/<member>`: one json per sample holding
    `{"source": ..., "texts": [...]}`, images under numbered keys (`0.jpg`, `1.jpg`, ...).

    Every `texts` entry is one user/assistant EXCHANGE, not one turn, so each yields two
    messages. Verified uniform across all 185 staged members (1480 samples: every json is
    `{source, texts}`, every turn is `{user, assistant}`); image count runs 0 (the `text_*`
    members) to 21 (drivelm), 1 for ~93%.
    """
    keys = sorted((k for k in sample if _NEMOTRON_IMG_KEY.match(k)),
                  key=lambda k: int(k.split(".", 1)[0]))
    images = [sample[k] for k in keys]

    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    for i, turn in enumerate(sample["json"]["texts"]):
        # 11 members carry a literal "<image>" in their text, but it is not a usable
        # position marker: drivelm names its 21 views in prose and carries none. So the
        # marker is stripped and every placeholder goes up front on the first user turn,
        # which is also what FineVision's own loader does.
        # ponytail: images-first; interleave only if a member is shown to need it.
        user = (turn.get("user") or "").replace("<image>", "").strip()
        assistant = (turn.get("assistant") or "").replace("<image>", "").strip()

        content = [{"type": "image"}] * len(images) if i == 0 else []
        content.append({"type": "text", "text": user})

        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant",
                         "content": [{"type": "text", "text": assistant}]})

    return EnergonSample(
        **basic_sample_keys(sample),
        image=images[0] if len(images) == 1 else None,
        images=images if len(images) > 1 else None,
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

_NEMOTRON_BARE_IMG_KEYS = ("png", "jpg", "jpeg", "webp")

@stateless
def cooker_nemotron(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """Nemotron-VLM-Dataset v1/v2/v3, repacked by `utils/prepare_nemotron_energon.py`"""
    keys = sorted((k for k in sample if _NEMOTRON_IMG_KEY.match(k)),
                  key=lambda k: int(k.split(".", 1)[0]))
    keys = keys or [k for k in _NEMOTRON_BARE_IMG_KEYS if k in sample]
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

DOCMAESTRO_PAGES_DIR = None  # set before COOKERS runs (see data/cookers.py header note)

DOCMAESTRO_DESC = {
    "basic_reading": "Read the page and transcribe its running text top to bottom.",
    "caption_prediction": "Given a figure's surrounding text blocks (no caption), predict the caption.",
    "doc_reconstruction": "Given the page's text/figure blocks (unordered), predict each block's bounding box.",
    "dom_edition": "Given the page rendered as a rough XML/DOM tree, correct it into the clean tagged tree.",
    "figure_ordering": "Given a page's figures out of order, predict the correct left-to-right/top-to-bottom order.",
    "figure_selection": "Given several figure crops (from different pages), pick the one belonging to this page.",
    "layout": "Given the page image, predict the bounding box + tag (title/text-block/figure/...) of every element.",
    "markdown": "Convert the page into clean Markdown.",
    "ocr": "Given the page image, transcribe every text block with its bounding box.",
    "page_sorting": "Given a document's pages shuffled, predict the correct page order.",
    "page_tree": "Convert the page into a structured tagged tree (title/text-block/figure with levels).",
    "reading_order": "Given a page's text/title blocks out of order, predict the correct reading order.",
}

_DOCMAESTRO_DECODE_IMG = _wds.autodecode.imagehandler("torchrgb")

def _docmaestro_text_of(sample: dict, key_txt: str, key_json) -> str:
    if key_txt in sample:
        return sample[key_txt]
    if key_json in sample:
        v = sample[key_json]
        return json.dumps(v) if not isinstance(v, str) else v
    return ""

@stateless
def cooker_docmaestro(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    meta = sample["json"]
    doc_id, page_idx = meta["doc_id"], meta["page_idx"]

    img_path = _Path(DOCMAESTRO_PAGES_DIR) / doc_id / f"p{page_idx:03d}.jpg"
    image = _DOCMAESTRO_DECODE_IMG("jpg", img_path.read_bytes())

    instruction = DOCMAESTRO_DESC[meta["task_type"]]
    input_text = _docmaestro_text_of(sample, "input.txt", "input.json")
    target_text = _docmaestro_text_of(sample, "target.txt", "target.json")

    user_text = f"{instruction}\n{input_text}".strip()

    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

_DCVLM_ROLE_MAP = {"human": "user", "gpt": "assistant"}

def _dcvlm_turn_content(text: str) -> list[dict]:
    parts = text.split("<image>")
    content = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            content.append({"type": "text", "text": part})
        if i < len(parts) - 1:
            content.append({"type": "image"})
    if not content:
        content.append({"type": "text", "text": ""})
    return content


@stateless
def cooker_dcvlm(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    for turn in sample["conversations.txt"].split("<EOCL>"):
        turn = turn.strip()
        if not turn:
            continue
        role_key, _, text = turn.partition(":")
        role = _DCVLM_ROLE_MAP.get(role_key.strip(), "user")
        messages.append({"role": role, "content": _dcvlm_turn_content(text)})

    img_keys = sorted((k for k in sample if _NEMOTRON_IMG_KEY.match(k)),
                       key=lambda k: int(k.split(".", 1)[0]))
    images = [sample[k] for k in img_keys]

    return EnergonSample(
        **basic_sample_keys(sample),
        image=images[0] if len(images) == 1 else None,
        images=images if len(images) > 1 else None,
        messages=messages,
    )

@stateless
def cooker_nemotron_v2(sample: dict, media_source, add_system_prompt: bool = True) -> EnergonSample:
    """Nemotron-VLM-Dataset-v2 (JSC/Jupiter `/p/data1/mmlaion/...`): same `{id, messages}`
    schema as v1/v3 (see `nemotron_messages`), but *polylithic* -- text lives in a plain
    `.jsonl` file, images in a separate prepared webdataset, joined by energon's
    MetadatasetV2 `aux: media_source:` mechanism (NVIDIA's `example_loader.py`
    `cook_conversation_polylithic`). The task encoder wraps every `aux` FileStore in a
    `DecodeFileStore` using its own `decoder` (`base.py:691`), so `media_source[key]`
    already returns a decoded image (PIL, matching `PackedBatchEncoder.decoder`) --
    no manual bytes decoding needed here, unlike `cooker_docmaestro`'s raw tar read.
    """
    return _polylithic(sample, media_source, add_system_prompt)

_ARXIV_PAGE = re.compile(r"^p-(\d+)\.png$")
DOCPARSE_INSTRUCTION = (
    "Parse this document page by page into clean Markdown, preserving reading order, "
    "headers, tables and equations (as LaTeX)."
)

@stateless
def cooker_arxiv_crude(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """MinerU-parsed arxiv papers straight from the crude webdataset
    (`long_document_arxiv_crude`: one sample per paper, `meta.json` + `p-NN.png/.md/.json`).

    Subflavor `docparse_view` = "turns" (default; one user image / assistant markdown
    pair per page, instruction on the first) or "whole" (every page in one turn, one
    target with "## Page N" separators). Whole pages, capped like every other dataset.
    The layout/native variants live in the doc-VLM checkout (qwen4-docvlm).
    """
    pid = sample["meta.json"]["paper_id"]
    pages = sorted((k for k in sample if _ARXIV_PAGE.match(k)), key=lambda k: int(_ARXIV_PAGE.match(k)[1]))
    refs = [f"{pid}/{p}" for p in pages]
    mds = [sample[p[:-4] + ".md"] for p in pages]  # no decoder for .md: raw bytes
    mds = [m.decode() if isinstance(m, bytes) else m for m in mds]

    if sample["__subflavors__"].get("docparse_view", "turns") == "whole":
        messages = [
            {"role": "user", "content": [{"type": "image", "image": r} for r in refs] + [DOCPARSE_INSTRUCTION]},
            {"role": "assistant", "content": ["\n\n".join(f"## Page {i + 1}\n\n{m}" for i, m in enumerate(mds))]},
        ]
    else:
        messages = []
        for i, (r, m) in enumerate(zip(refs, mds)):
            messages.append({"role": "user", "content": [{"type": "image", "image": r}] + [DOCPARSE_INSTRUCTION] * (i == 0)})
            messages.append({"role": "assistant", "content": [m]})
    media = {r: (sample[p], None) for r, p in zip(refs, pages)}
    return _polylithic({**sample, "json": {"messages": messages}}, media, add_system_prompt)

def _polylithic(sample: dict, media_source, add_system_prompt: bool) -> EnergonSample:
    data = sample["json"]
    images = []
    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    for turn in data["messages"]:
        content = []
        for part in turn["content"]:
            if isinstance(part, str):
                if part:
                    content.append({"type": "text", "text": part})
            elif part.get("type") == "image":
                image, _src = media_source[part["image"]]
                images.append(image)
                content.append({"type": "image"})
            elif part.get("text"):
                content.append({"type": "text", "text": part["text"]})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": turn["role"], "content": content})

    return EnergonSample(
        **basic_sample_keys(sample),
        image=images[0] if len(images) == 1 else None,
        images=images if len(images) > 1 else None,
        messages=messages,
    )

COOKERS = [
    # subflavors can be used to distinguish datasets when using a Metadataset
    Cooker(cooker_nemotron_v2, has_subflavors={"type_dataset": "ccpdf"}),
    Cooker(cooker_nemotron_v2, has_subflavors={"type_dataset": "nemotron_v2"}),
    Cooker(cooker_docmaestro, has_subflavors={"type_dataset": "docmaestro"}),
    Cooker(cooker_dcvlm, has_subflavors={"type_dataset": "dcvlm"}),
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
    # long_document_arxiv_docparse (build_docparse_jsonl.py, MinerU2.5-Pro-parsed arxiv
    # pages): same polylithic {id, messages} schema again. Two views over the same page
    # images -- whole: one shot, every page + full-doc markdown target; turns: one
    # image/markdown turn pair per page in a growing conversation. Both just walk
    # messages generically, same cooker, no page cap (whole paper is always one sample).
    Cooker(cooker_nemotron_v2, has_subflavors={"type_dataset": "docparse_whole"}),
    Cooker(cooker_nemotron_v2, has_subflavors={"type_dataset": "docparse_turns"}),
    Cooker(cooker_arxiv_crude, has_subflavors={"type_dataset": "arxiv_docparse"}),
]

import json
import re

from megatron.energon import Cooker, basic_sample_keys, stateless
from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample

import io
import os

from PIL import Image

from data import docmaestro as _docmaestro

import torch

_NEMOTRON_IMG_KEY = re.compile(r"^(\d+)\.\w+$")

@edataclass
class EnergonSample(Sample):
    image: torch.Tensor | None
    messages: list
    images: list | None = None
    prompt_version: str | None = None  # DocMaestro prompt set the sample was asked with

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

# DocMaestro (VLR-CVC/DocMaestro) task shards. Paths and prompt set come from the
# dataset's subflavors (`docmaestro_pages_dir`, `docmaestro_crops_root`,
# `docmaestro_prompts`, `docmaestro_prompt_seed`), falling back to these globals, which
# read the same names upper-cased from the environment. See docs/USAGE.md "DocMaestro".
DOCMAESTRO_PAGES_DIR = os.environ.get("DOCMAESTRO_PAGES_DIR")    # page store: <doc_id>/pNNN.jpg
DOCMAESTRO_CROPS_ROOT = os.environ.get("DOCMAESTRO_CROPS_ROOT")  # extraction tree: figure crops
DOCMAESTRO_PROMPTS = os.environ.get("DOCMAESTRO_PROMPTS", str(_docmaestro.DEFAULT_PROMPTS_PATH))
DOCMAESTRO_PROMPT_SEED = int(os.environ.get("DOCMAESTRO_PROMPT_SEED", "0"))

# One fixed instruction per task: used for shards that predate `assets.json`, and for
# new ones when the prompt set is "fixed".
DOCMAESTRO_DESC = {
    "basic_reading": "Read the page and transcribe its running text top to bottom.",
    "caption_prediction": "Given a figure's surrounding text blocks (no caption), predict the caption.",
    "doc_reconstruction": "Given the page's text/figure blocks (unordered), predict each block's bounding box.",
    "dom_edition": "Given the page rendered as a rough XML/DOM tree, correct it into the clean tagged tree.",
    "figure_ordering": "Given a document's figures and some of its pages, map each figure to the page it came from.",
    "figure_selection": "Given several figure crops (from different pages), pick the one belonging to this page.",
    "form_filling": "Extract all form fields on this page with their category and bounding box as JSON.",
    "layout": "Given the page image, predict the bounding box + tag (title/text-block/figure/...) of every element.",
    "markdown": "Convert the page into clean Markdown.",
    "ocr": "Given the page image, transcribe every text block with its bounding box.",
    "page_sorting": "Given a document's pages shuffled, predict the correct page order.",
    "page_tree": "Convert the page into a structured tagged tree (title/text-block/figure with levels).",
    "reading_order": "Given a page's text/title blocks out of order, predict the correct reading order.",
}

def _docmaestro_decode(raw: bytes):
    """PIL RGB, what `PackedBatchEncoder` (decoder `image_decode="pil"`) works on."""
    return Image.open(io.BytesIO(raw)).convert("RGB")

def _docmaestro_text_of(sample: dict, key_txt: str, key_json) -> str:
    if key_txt in sample:
        return sample[key_txt]
    if key_json in sample:
        v = sample[key_json]
        return json.dumps(v) if not isinstance(v, str) else v
    return ""

def _docmaestro_field(sample: dict, name: str):
    """`<name>.json` parsed, or `<name>.txt` as str, or None. Energon decodes both by
    extension already; raw bytes are accepted too."""
    for ext, parse in (("json", json.loads), ("txt", lambda b: b)):
        if f"{name}.{ext}" in sample:
            v = sample[f"{name}.{ext}"]
            if isinstance(v, (bytes, bytearray)):
                v = parse(v.decode("utf-8"))
            return v
    return None

def _docmaestro_messages(content: list, target_text: str, add_system_prompt: bool) -> list:
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
    ]
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
    return messages

@stateless
def cooker_docmaestro(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """A DocMaestro task sample, fed exactly the images its task contract names.

    `assets.json` lists page renders (from the page store) and figure crops (from the
    extraction tree); `data/docmaestro.py` checks them against the task's contract
    and raises `AssetArityError` on a mismatch. `caption_prediction` and
    `figure_selection` get crops only and `doc_reconstruction` no image: the page
    shows their answers. `page_sorting` pages have their page numbers blacked out.
    Images come first, in `assets.json` order (pages, then figures), labelled
    "Page i:" / "Figure i:" when there are several, then the prompt: one of the
    task's DocMaestro templates, chosen by (task, key, seed).

    Shards without `assets.json` (written before DocMaestro had it) keep the old
    one-page behaviour.
    """
    cfg = sample.get("__subflavors__") or {}
    meta = sample["json"]
    task = meta["task_type"]
    pages_dir = cfg.get("docmaestro_pages_dir") or DOCMAESTRO_PAGES_DIR
    target_text = _docmaestro_text_of(sample, "target.txt", "target.json")

    assets = _docmaestro_field(sample, "assets")
    if assets is None:
        return _cooker_docmaestro_legacy(sample, meta, pages_dir, target_text, add_system_prompt)

    _docmaestro.validate_asset_arity(task, assets)
    contract = _docmaestro.get_contract(task)
    input_value = _docmaestro_field(sample, "input")

    pages = [_docmaestro_decode(_docmaestro.read_asset(_docmaestro.page_path(pages_dir, r["doc_id"], r["page_idx"])))
             for r in assets["page_images"]]
    crops_root = cfg.get("docmaestro_crops_root") or DOCMAESTRO_CROPS_ROOT
    figures = [_docmaestro_decode(_docmaestro.read_asset(_docmaestro.figure_path(crops_root, r["doc_id"], r["src"])))
               for r in assets["figure_images"]]

    if contract.redact_page_numbers:
        # input entry i describes page image i: never reorder either list
        entries = input_value if isinstance(input_value, list) else []
        pages = [_docmaestro.redact_boxes(p, entries[i].get("page_number_bboxes", []) if i < len(entries) else [])
                 for i, p in enumerate(pages)]

    prompts = cfg.get("docmaestro_prompts") or DOCMAESTRO_PROMPTS
    if prompts == "fixed":
        prompt_version = "fixed"
        text = DOCMAESTRO_DESC[task]
        if contract.text_input and input_value is not None:
            text = f"{text}\n{_docmaestro._dumps(input_value)}"
    else:
        registry = _docmaestro.load_prompts(str(prompts))
        prompt_version = registry.version
        seed = int(cfg.get("docmaestro_prompt_seed", DOCMAESTRO_PROMPT_SEED))
        fields = _docmaestro.prompt_fields(task, input_value, len(pages), len(figures))
        text = registry.render(task, sample["__key__"], seed, fields)
        if contract.text_input and input_value is not None and not registry.placeholders(task):
            text = f"{text}\n\n{_docmaestro._dumps(input_value)}"  # a template set that forgot the payload

    page_labels, figure_labels = _docmaestro.image_labels(task, input_value, len(pages), len(figures))
    content = []
    for label in page_labels + figure_labels:
        if label is not None:
            content.append({"type": "text", "text": label})
        content.append({"type": "image"})
    content.append({"type": "text", "text": text})

    images = pages + figures
    return EnergonSample(
        **basic_sample_keys(sample),
        image=images[0] if len(images) == 1 else None,
        images=images if len(images) > 1 else None,
        messages=_docmaestro_messages(content, target_text, add_system_prompt),
        prompt_version=prompt_version,
    )

def _cooker_docmaestro_legacy(sample, meta, pages_dir, target_text, add_system_prompt) -> EnergonSample:
    """Pre-`assets.json` shards: the page `meta.page_idx`, a fixed instruction, the input."""
    task = meta["task_type"]
    if "page_idx" not in meta:
        raise _docmaestro.AssetArityError(
            f"{sample['__key__']}: {task} shard has no assets.json and no page_idx; it needs a "
            "DocMaestro shard written with asset references")
    img_path = _docmaestro.page_path(pages_dir, meta["doc_id"], meta["page_idx"])
    image = _docmaestro_decode(_docmaestro.read_asset(img_path))

    input_text = _docmaestro_text_of(sample, "input.txt", "input.json")
    user_text = f"{DOCMAESTRO_DESC[task]}\n{input_text}".strip()
    content = [{"type": "image"}, {"type": "text", "text": user_text}]
    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=_docmaestro_messages(content, target_text, add_system_prompt),
        prompt_version="fixed",
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
    """Nemotron-VLM-Dataset-v2 (JSC/Jupiter `/p/data1/mmlaion/...`)`"""
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
    Cooker(cooker_nemotron_v2, has_subflavors={"type_dataset": "docparse_whole"}),
    Cooker(cooker_nemotron_v2, has_subflavors={"type_dataset": "docparse_turns"}),
    Cooker(cooker_arxiv_crude, has_subflavors={"type_dataset": "arxiv_docparse"}),
]

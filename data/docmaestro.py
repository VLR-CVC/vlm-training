"""DocMaestro shards, read the way DocMaestro's own loader reads them.

A DocMaestro sample carries asset *references* (`assets.json`), never pixels, and each
task has a contract saying how many page renders and figure crops it gets. Three tasks
get no page on purpose: `caption_prediction` and `figure_selection` would find their
answer printed on it, and `doc_reconstruction` would read its target coordinates off
it. This module is the vlm-training side of that contract; `cooker_docmaestro` in
`data/cookers.py` is the only caller.

Small pure pieces are vendored from DocMaestro @ fa00f2e (`src/doc_maestro/training/`)
rather than imported, because DocMaestro pulls in `mineru[all]`:

| here | DocMaestro |
|---|---|
| `CONTRACTS`, `validate_asset_arity` | `contract.py` |
| `page_path`, `figure_path` (crop-dir lookup) | `rendering/store.py`, `assets.py` |
| `redact_boxes` | `transforms.py` (draws on the decoded image: no JPEG round trip) |
| `PromptRegistry.select` | `prompts.py`, same digest, `epoch` -> `seed` |

`models/tests/test_cooker_docmaestro.py` pins the table against a DocMaestro checkout
when one is available.
"""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import yaml
from PIL import Image, ImageDraw

NONE, SINGLE, MULTI = "none", "single", "multi"


@dataclass(frozen=True)
class Contract:
    page_images: str            # NONE / SINGLE / MULTI
    figure_images: str
    text_input: bool            # the prompt carries a structured payload (input.*)
    redact_page_numbers: bool = False


# DocMaestro training/contract.py, field for field. The NONE entries are leak
# avoidance, not omissions: see the module docstring.
CONTRACTS: dict[str, Contract] = {
    "basic_reading": Contract(SINGLE, NONE, False),
    "ocr": Contract(SINGLE, NONE, False),
    "layout": Contract(SINGLE, NONE, False),
    "markdown": Contract(SINGLE, NONE, False),
    "page_tree": Contract(SINGLE, NONE, False),
    "form_filling": Contract(SINGLE, NONE, False),
    "dom_edition": Contract(SINGLE, NONE, True),
    "reading_order": Contract(SINGLE, NONE, True),
    "doc_reconstruction": Contract(NONE, NONE, True),
    "caption_prediction": Contract(NONE, SINGLE, True),
    "figure_selection": Contract(NONE, MULTI, True),
    "page_sorting": Contract(MULTI, NONE, False, redact_page_numbers=True),
    "figure_ordering": Contract(MULTI, MULTI, True),
}


class AssetArityError(ValueError):
    """A sample's `assets.json` does not match its task contract. Never skipped: it
    means the composer is wrong, and training on it either leaks the answer or
    trains on blank input."""


class MissingAsset(FileNotFoundError):
    """A referenced page render or figure crop is not on disk."""


def get_contract(task: str) -> Contract:
    try:
        return CONTRACTS[task]
    except KeyError:
        raise KeyError(f"no DocMaestro contract for task {task!r}; known: {sorted(CONTRACTS)}") from None


def validate_asset_arity(task: str, assets: dict) -> None:
    if set(assets) != {"page_images", "figure_images"}:
        raise AssetArityError(f"{task}: assets.json keys must be exactly "
                              f"['figure_images', 'page_images'], got {sorted(assets)}")
    contract = get_contract(task)
    for field, policy in (("page_images", contract.page_images),
                          ("figure_images", contract.figure_images)):
        n = len(assets[field])
        if (policy == NONE and n) or (policy == SINGLE and n != 1) or (policy == MULTI and n < 1):
            raise AssetArityError(f"{task}.{field}: contract says {policy.upper()}, got {n}")


# ---- asset layout -----------------------------------------------------------------

def page_path(pages_root, doc_id: str, page_idx: int) -> Path:
    """`<pages_root>/<VENUE>/<YEAR>/<stem>/pNNN.jpg`, DocMaestro's page store."""
    return Path(pages_root).joinpath(*doc_id.split("/")) / f"p{int(page_idx):03d}.jpg"


@functools.lru_cache(maxsize=65536)
def _crops_dir(crops_root: str, doc_id: str) -> Path | None:
    """The directory `<img src>` resolves against: the document folder itself if it
    holds `images/`, else its first child that does (MinerU's `hybrid_auto/`). One
    document's folder only, never a corpus-wide walk, which is slow and can return
    another document's crops. Misses are cached too."""
    doc_dir = Path(crops_root).joinpath(*doc_id.split("/"))
    if (doc_dir / "images").is_dir():
        return doc_dir
    if not doc_dir.is_dir():
        return None
    for child in sorted(doc_dir.iterdir()):
        if child.is_dir() and (child / "images").is_dir():
            return child
    return None


def figure_path(crops_root, doc_id: str, src: str) -> Path:
    if crops_root is None:
        raise MissingAsset(f"no DocMaestro crops root configured; cannot resolve {src!r} for {doc_id!r}")
    root = _crops_dir(str(crops_root), doc_id)
    if root is None or not src:
        raise MissingAsset(f"no crops directory for {doc_id!r} under {crops_root}")
    return root / src


def read_asset(path: Path) -> bytes:
    if not path.is_file():
        raise MissingAsset(f"DocMaestro asset not on disk: {path}")
    return path.read_bytes()


# ---- redaction ----------------------------------------------------------------------

BBOX_SPACE = 1000.0


def parse_bbox(raw) -> list[int]:
    """XML attributes carry `"[x1, y1, x2, y2]"`, decoded JSON a list: accept both."""
    values = json.loads(raw) if isinstance(raw, str) else raw
    return [int(v) for v in values]


def redact_boxes(image: Image.Image, bboxes, fill=(0, 0, 0)) -> Image.Image:
    """Black out `[0, 1000]`-normalised boxes. Returns the input untouched when there
    is nothing to redact, a redacted copy otherwise."""
    if not bboxes:
        return image
    image = image.copy()
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for bbox in bboxes:
        x1, y1, x2, y2 = parse_bbox(bbox)
        draw.rectangle((int(x1 / BBOX_SPACE * w), int(y1 / BBOX_SPACE * h),
                        int(x2 / BBOX_SPACE * w), int(y2 / BBOX_SPACE * h)), fill=fill)
    return image


# ---- prompts ------------------------------------------------------------------------

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "assets" / "docmaestro_doctasks.yaml"


@dataclass(frozen=True)
class PromptRegistry:
    version: str
    tasks: dict

    @classmethod
    def load(cls, path) -> "PromptRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not raw.get("version") or not raw.get("tasks"):
            raise ValueError(f"prompt registry {path} needs top-level 'version' and 'tasks'")
        tasks = {}
        for name, templates in raw["tasks"].items():
            if not templates:
                raise ValueError(f"task {name!r} in {path} has no prompt templates")
            tasks[name] = tuple(templates)
        return cls(str(raw["version"]), tasks)

    def placeholders(self, task: str) -> set[str]:
        return {f for t in self.tasks[task] for _, f, _, _ in Formatter().parse(t) if f}

    def select(self, task: str, key: str, seed: int = 0) -> str:
        """DocMaestro's `PromptRegistry.select` with `epoch` -> `seed`: a pure function
        of (task, key, seed), so workers, restarts and machines agree without state."""
        templates = self.tasks[task]
        digest = hashlib.sha1(f"{task}|{key}|{seed}".encode("utf-8")).digest()
        return templates[int.from_bytes(digest[:8], "big") % len(templates)]

    def render(self, task: str, key: str, seed: int, fields: dict[str, Any]) -> str:
        template = self.select(task, key, seed)
        try:
            return template.format(**fields)
        except KeyError as exc:
            raise KeyError(f"DocMaestro prompt for {task!r} needs field {exc.args[0]!r}; "
                           f"template: {template!r}") from None


@functools.lru_cache(maxsize=8)
def load_prompts(path: str) -> PromptRegistry:
    return PromptRegistry.load(path)


def _dumps(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def prompt_fields(task: str, input_value, n_pages: int, n_figures: int) -> dict[str, Any]:
    """Every placeholder `doctasks.yaml` uses, filled from the sample. DocMaestro
    names the fields but leaves the mapping to the loader; this is that mapping.
    A superset is fine: `str.format` ignores extra keys."""
    fields: dict[str, Any] = {"n_pages": n_pages, "n_figures": n_figures, "n_candidates": n_figures}
    if input_value is None:
        return fields
    text = _dumps(input_value)
    fields.update(tree=text, elements=text, context=text, page_text=text)
    if task == "figure_ordering" and isinstance(input_value, dict):
        fields["pages"] = _dumps(input_value.get("pages", []))
    return fields


def image_labels(task: str, input_value, n_pages: int, n_figures: int) -> tuple[list, list]:
    """Text put before each image in multi-image tasks, so the indices in the prompt
    and in the target can be tied to an image: `page_sorting` and `figure_selection`
    answer with presented positions, `figure_ordering` with real page indices.
    Single-image tasks get no label (unchanged layout)."""
    if n_pages + n_figures <= 1:
        return [None] * n_pages, [None] * n_figures
    pages = list(range(n_pages))
    if task == "figure_ordering" and isinstance(input_value, dict):
        listed = input_value.get("pages") or []
        if len(listed) == n_pages:
            pages = listed
    return [f"Page {p}:" for p in pages], [f"Figure {i}:" for i in range(n_figures)]

"""`cooker_docmaestro` feeds every DocMaestro task exactly the images its contract names.

DocMaestro shards carry asset *references* (`assets.json`). The old cooker ignored them
and always loaded the sample's page render, which turns `caption_prediction` and
`figure_selection` into OCR (the answer is printed on the page) and cannot feed the
multi-image tasks at all.

Two layers:
- synthetic: a tiny page store + crop tree written to `tmp_path`, one sample per task
  built the way DocMaestro's composers write them. Always runs, CPU only.
- real: shards composed by DocMaestro itself on two CVPR papers, through the cooker and
  `PackedBatchEncoder.encode_sample`. Needs the fixture (`DOCMAESTRO_FIXTURE`, built by
  `utils/build_docmaestro_fixture.py`) and, for the encoder, a Qwen3.5 processor
  (`DOCMAESTRO_TEST_PROCESSOR`, default the HF cache's `Qwen/Qwen3.5-0.8B`). Skipped
  when either is missing.
"""
import collections
import glob
import json
import os
import tarfile
from pathlib import Path

import pytest
from PIL import Image

from data import docmaestro as dm
from data.cookers import DOCMAESTRO_DESC, cooker_docmaestro

TASKS = sorted(dm.CONTRACTS)
DOC = "CVPR/2020/Some.Paper_v1.5"   # dots in the stem: DocMaestro's path hazard
PAGE_SIZE = (100, 130)
N_PAGES = 6
N_FIGS = 4


def _fig_size(k):
    return (60 + k, 40)   # every crop has its own size, and none is a page's


@pytest.fixture
def store(tmp_path):
    pages, crops = tmp_path / "pages", tmp_path / "processed"
    for i in range(N_PAGES):
        p = dm.page_path(pages, DOC, i)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", PAGE_SIZE, (255, 255, 255)).save(p, quality=95)
    img_dir = crops.joinpath(*DOC.split("/")) / "hybrid_auto" / "images"
    img_dir.mkdir(parents=True)
    for k in range(N_FIGS):
        Image.new("RGB", _fig_size(k), (0, 128, 255)).save(img_dir / f"fig{k}.jpg")
    return {"type_dataset": "docmaestro", "docmaestro_pages_dir": str(pages),
            "docmaestro_crops_root": str(crops)}


def _page(i):
    return {"doc_id": DOC, "page_idx": i}


def _fig(k):
    return {"doc_id": DOC, "src": f"images/fig{k}.jpg"}


PAGE_NUMBER_BOX = "[450, 950, 550, 990]"

# (assets, input) per task, shaped like DocMaestro's composers emit them
SPECS = {
    "basic_reading": ([_page(0)], [], None),
    "ocr": ([_page(1)], [], None),
    "layout": ([_page(1)], [], None),
    "markdown": ([_page(2)], [], None),
    "page_tree": ([_page(2)], [], None),
    "form_filling": ([_page(0)], [], None),
    "dom_edition": ([_page(3)], [], "<page idx=\"3\"><title>x</title></page>"),
    "reading_order": ([_page(3)], [], [{"idx": 0, "tag": "title", "text": "Intro"}]),
    "doc_reconstruction": ([], [], [{"idx": 0, "tag": "title", "text": "Intro"}]),
    "caption_prediction": ([], [_fig(2)], {"preceding": [], "following": []}),
    "figure_selection": ([], [_fig(3), _fig(0), _fig(1)], "page text"),
    "page_sorting": ([_page(4), _page(2), _page(3)], [],
                     [{"idx": i, "page_number_bboxes": [PAGE_NUMBER_BOX]} for i in range(3)]),
    "figure_ordering": ([_page(0), _page(1), _page(5)], [_fig(1), _fig(0), _fig(3)],
                        {"pages": [0, 1, 5], "figures": [{"idx": 0, "src": "images/fig1.jpg"}]}),
}


def _sample(task, store, assets=None, inp=None, key=None, **sub):
    pages, figs, default_inp = SPECS[task]
    meta = {"task_type": task, "doc_name": DOC.rsplit("/", 1)[1], "doc_id": DOC}
    s = {"__key__": key or f"k_{task}", "__restore_key__": (), "__sources__": (),
         "__subflavors__": dict(store, **sub), "json": meta,
         "assets.json": assets if assets is not None else {"page_images": pages, "figure_images": figs},
         "target.txt": "the answer"}
    inp = default_inp if inp is None else inp
    if isinstance(inp, str):
        s["input.txt"] = inp
    elif inp is not None:
        s["input.json"] = inp
    return s


def _images(cooked):
    return cooked.images or ([cooked.image] if cooked.image is not None else [])


def _placeholders(cooked):
    return sum(part["type"] == "image" for m in cooked.messages for part in m["content"])


# ---- every task, synthetic ---------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_each_task_gets_exactly_its_contract(task, store):
    pages, figs, _ = SPECS[task]
    cooked = cooker_docmaestro(_sample(task, store))
    imgs = _images(cooked)
    assert len(imgs) == len(pages) + len(figs)
    assert _placeholders(cooked) == len(imgs)
    # one image -> `image`, several -> `images`, as cooker_nemotron does
    assert (cooked.image is not None) == (len(imgs) == 1)
    assert (cooked.images is not None) == (len(imgs) > 1)
    # sources, in order: pages first, then figures
    want = [PAGE_SIZE] * len(pages) + [_fig_size(int(f["src"][-5])) for f in figs]
    assert [im.size for im in imgs] == want
    assert all(isinstance(im, Image.Image) and im.mode == "RGB" for im in imgs)
    if dm.CONTRACTS[task].page_images == dm.NONE:
        assert PAGE_SIZE not in [im.size for im in imgs], f"{task} must never see a page"


def test_caption_prediction_gets_the_figure_crop_never_the_page(store):
    """Regression: the old cooker loaded `pages/<doc>/p<page_idx>.jpg` for every task,
    and the caption is printed on that page."""
    s = _sample("caption_prediction", store)
    s["json"]["page_idx"] = 0     # what the real shard carries, and what the old cooker read
    cooked = cooker_docmaestro(s)
    assert cooked.images is None and cooked.image is not None
    assert cooked.image.size == _fig_size(2) != PAGE_SIZE
    assert cooked.image.getpixel((5, 5))[2] > 200   # the blue crop, not the white page


@pytest.mark.parametrize("task,pages,figs", [
    ("caption_prediction", [_page(0)], [_fig(2)]),     # page leak
    ("caption_prediction", [], []),                    # nothing to caption
    ("figure_selection", [_page(0)], [_fig(1), _fig(2)]),
    ("doc_reconstruction", [_page(0)], []),
    ("ocr", [_page(0), _page(1)], []),
    ("page_sorting", [], []),
    ("figure_ordering", [_page(0)], []),
])
def test_arity_mismatch_fails_loudly(task, pages, figs, store):
    with pytest.raises(dm.AssetArityError):
        cooker_docmaestro(_sample(task, store, assets={"page_images": pages, "figure_images": figs}))


def test_missing_asset_fails_loudly(store):
    with pytest.raises(dm.MissingAsset):
        cooker_docmaestro(_sample("ocr", store, assets={"page_images": [_page(99)], "figure_images": []}))
    with pytest.raises(dm.MissingAsset):
        cooker_docmaestro(_sample("caption_prediction", store,
                                  assets={"page_images": [], "figure_images": [{"doc_id": DOC, "src": "images/nope.jpg"}]}))


def test_assets_json_as_raw_bytes(store):
    """Energon decodes `.json` members already; raw bytes must work too."""
    s = _sample("caption_prediction", store)
    s["assets.json"] = json.dumps(s["assets.json"]).encode()
    assert cooker_docmaestro(s).image.size == _fig_size(2)


# ---- page_sorting redaction ---------------------------------------------------------

def test_page_sorting_blacks_out_page_numbers(store):
    cooked = cooker_docmaestro(_sample("page_sorting", store))
    for im in cooked.images:
        w, h = im.size
        cx, cy = int(0.5 * w), int(0.97 * h)              # centre of PAGE_NUMBER_BOX
        assert im.getpixel((cx, cy)) == (0, 0, 0)
        assert min(im.getpixel((5, 5))) > 240              # the rest stays white


def test_redaction_follows_its_own_page(store):
    """input entry i carries page image i's boxes; only that page is redacted there."""
    inp = [{"idx": 0, "page_number_bboxes": []},
           {"idx": 1, "page_number_bboxes": [PAGE_NUMBER_BOX]},
           {"idx": 2, "page_number_bboxes": []}]
    imgs = cooker_docmaestro(_sample("page_sorting", store, inp=inp)).images
    probe = [im.getpixel((int(0.5 * im.size[0]), int(0.97 * im.size[1]))) for im in imgs]
    assert probe[1] == (0, 0, 0)
    assert min(probe[0]) > 240 and min(probe[2]) > 240


def test_page_sorting_prompt_carries_no_input():
    """Its input is redaction metadata; the bboxes do not belong in the prompt."""
    reg = dm.load_prompts(str(dm.DEFAULT_PROMPTS_PATH))
    assert reg.placeholders("page_sorting") == {"n_pages"}
    assert not dm.CONTRACTS["page_sorting"].text_input


# ---- labels -------------------------------------------------------------------------

def test_multi_image_labels(store):
    content = cooker_docmaestro(_sample("figure_ordering", store)).messages[0]["content"]
    labels = [p["text"] for p in content if p["type"] == "text"][:-1]
    # figure_ordering's target names real page indices, so pages carry those
    assert labels == ["Page 0:", "Page 1:", "Page 5:", "Figure 0:", "Figure 1:", "Figure 2:"]
    content = cooker_docmaestro(_sample("figure_selection", store)).messages[0]["content"]
    assert [p["text"] for p in content if p["type"] == "text"][:-1] == ["Figure 0:", "Figure 1:", "Figure 2:"]
    # single image: no label, image then prompt, as before
    content = cooker_docmaestro(_sample("ocr", store)).messages[0]["content"]
    assert [p["type"] for p in content] == ["image", "text"]


# ---- prompts ------------------------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_prompt_is_a_rendered_template(task, store):
    reg = dm.load_prompts(str(dm.DEFAULT_PROMPTS_PATH))
    cooked = cooker_docmaestro(_sample(task, store))
    text = cooked.messages[0]["content"][-1]["text"]
    assert cooked.prompt_version == reg.version == "doctasks-1"
    heads = [t.split("{", 1)[0] for t in reg.tasks[task]]
    assert any(text.startswith(h) for h in heads), text
    assert "{" not in text.split("\n")[0], f"unfilled placeholder: {text!r}"
    _, _, inp = SPECS[task]
    if dm.CONTRACTS[task].text_input:
        payload = inp if isinstance(inp, str) else None
        if task == "figure_ordering":
            payload = "[0, 1, 5]"
        elif payload is None:
            payload = json.dumps(inp)
        assert payload in text


def test_prompt_choice_is_deterministic_and_varies(store):
    a = cooker_docmaestro(_sample("markdown", store, key="same"))
    b = cooker_docmaestro(_sample("markdown", store, key="same"))
    assert a.messages == b.messages
    seen = {cooker_docmaestro(_sample("markdown", store, key=f"k{i}")).messages[0]["content"][-1]["text"]
            for i in range(40)}
    assert len(seen) == 4      # all four paraphrases show up across samples
    seeds = {cooker_docmaestro(_sample("markdown", store, key="same", docmaestro_prompt_seed=s))
             .messages[0]["content"][-1]["text"] for s in range(20)}
    assert len(seeds) > 1


def test_same_choice_as_docmaestro():
    """`select(task, key, seed)` == DocMaestro's `select(task, sample_key=key, epoch=seed)`."""
    reg = dm.load_prompts(str(dm.DEFAULT_PROMPTS_PATH))
    import hashlib
    for key in ("a", "b_p003_ocr_0"):
        d = hashlib.sha1(f"ocr|{key}|7".encode()).digest()
        assert reg.select("ocr", key, 7) == reg.tasks["ocr"][int.from_bytes(d[:8], "big") % 4]


def test_fixed_prompts(store):
    cooked = cooker_docmaestro(_sample("reading_order", store, docmaestro_prompts="fixed"))
    text = cooked.messages[0]["content"][-1]["text"]
    assert text.startswith(DOCMAESTRO_DESC["reading_order"]) and '"Intro"' in text
    assert cooked.prompt_version == "fixed"


def test_every_task_has_a_fixed_description_and_templates():
    reg = dm.load_prompts(str(dm.DEFAULT_PROMPTS_PATH))
    assert set(DOCMAESTRO_DESC) == set(dm.CONTRACTS) == set(reg.tasks)


def test_prompts_path_from_subflavor(store, tmp_path):
    custom = tmp_path / "p.yaml"
    custom.write_text("version: test-9\ntasks:\n  ocr: ['Just OCR it.']\n")
    cooked = cooker_docmaestro(_sample("ocr", store, docmaestro_prompts=str(custom)))
    assert cooked.messages[0]["content"][-1]["text"] == "Just OCR it."
    assert cooked.prompt_version == "test-9"


# ---- backward compatibility --------------------------------------------------------

@pytest.mark.parametrize("task", ["ocr", "reading_order", "dom_edition"])
def test_shards_without_assets_json_keep_the_old_behaviour(task, store):
    s = _sample(task, store)
    del s["assets.json"]
    s["json"]["page_idx"] = 3
    cooked = cooker_docmaestro(s)
    # exactly what the old cooker built: page meta.page_idx, fixed instruction + input
    inp = s.get("input.txt") or (json.dumps(s["input.json"]) if "input.json" in s else "")
    assert cooked.messages == [
        {"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": f"{DOCMAESTRO_DESC[task]}\n{inp}".strip()}]},
        {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
    ]
    assert cooked.images is None and cooked.image.size == PAGE_SIZE
    assert cooked.prompt_version == "fixed"


def test_old_shard_without_page_idx_says_why(store):
    s = _sample("page_sorting", store)
    del s["assets.json"]
    with pytest.raises(dm.AssetArityError, match="assets.json"):
        cooker_docmaestro(s)


def test_pages_dir_from_global_when_no_subflavor(store, monkeypatch):
    import data.cookers as cookers
    monkeypatch.setattr(cookers, "DOCMAESTRO_PAGES_DIR", store["docmaestro_pages_dir"])
    s = _sample("ocr", store)
    s["__subflavors__"] = {"type_dataset": "docmaestro"}
    assert cooker_docmaestro(s).image.size == PAGE_SIZE


# ---- pinned against DocMaestro when a checkout is around ---------------------------

DOCMAESTRO = Path(os.environ.get("DOCMAESTRO_SRC", os.path.expanduser("~/DocMaestro")))


@pytest.mark.skipif(not (DOCMAESTRO / "src/doc_maestro/training/contract.py").is_file(),
                    reason="no DocMaestro checkout (DOCMAESTRO_SRC)")
def test_vendored_pieces_match_docmaestro():
    import ast
    tree = ast.parse((DOCMAESTRO / "src/doc_maestro/training/contract.py").read_text())
    theirs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "TaskContract":
            name = node.args[0].value
            pol = [a.attr.lower() for a in node.args[1:3]]
            kw = {k.arg: k.value for k in node.keywords}
            theirs[name] = dm.Contract(pol[0], pol[1], kw["text_input"].value,
                                       kw["redact_page_numbers"].value if "redact_page_numbers" in kw else False)
    assert theirs == dm.CONTRACTS
    import yaml
    ours = yaml.safe_load(dm.DEFAULT_PROMPTS_PATH.read_text())
    upstream = yaml.safe_load((DOCMAESTRO / "configs/prompts/doctasks.yaml").read_text())
    assert ours == upstream, "assets/docmaestro_doctasks.yaml drifted from DocMaestro's"


# ---- real DocMaestro shards -------------------------------------------------------

FIXTURE = Path(os.environ.get("DOCMAESTRO_FIXTURE", "/data/151-1/users/allabres/t11_docmaestro_fixture"))
needs_fixture = pytest.mark.skipif(not (FIXTURE / "shards").is_dir(),
                                   reason=f"no DocMaestro fixture at {FIXTURE} (utils/build_docmaestro_fixture.py)")


def _real_samples():
    """{task: first decoded sample}, decoded by energon's own SampleDecoder."""
    from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
    decoder = SampleDecoder(image_decode="pil")
    sub = {"type_dataset": "docmaestro", "docmaestro_pages_dir": str(FIXTURE / "pages"),
           "docmaestro_crops_root": str(FIXTURE / "processed")}
    out = {}
    for tar in sorted(glob.glob(str(FIXTURE / "shards" / "*" / "*.tar"))):
        groups = collections.OrderedDict()
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                key, ext = m.name.split(".", 1)
                groups.setdefault(key, {})[ext] = tf.extractfile(m).read()
        for key, parts in groups.items():
            raw = {"__key__": key, "__restore_key__": (), "__sources__": (), "__subflavors__": sub, **parts}
            out.setdefault(json.loads(parts["json"])["task_type"], decoder(raw))
    return out


@pytest.fixture(scope="module")
def real():
    return _real_samples()


@needs_fixture
def test_real_fixture_covers_all_tasks(real):
    assert set(real) == set(dm.CONTRACTS)


@needs_fixture
@pytest.mark.parametrize("task", TASKS)
def test_real_sample_images_match_assets(task, real):
    s = real[task]
    cooked = cooker_docmaestro(s)
    imgs = _images(cooked)
    assets = s["assets.json"]
    assert len(imgs) == len(assets["page_images"]) + len(assets["figure_images"])
    assert _placeholders(cooked) == len(imgs)
    pages = [Image.open(dm.page_path(FIXTURE / "pages", r["doc_id"], r["page_idx"])).convert("RGB")
             for r in assets["page_images"]]
    figs = [Image.open(dm.figure_path(FIXTURE / "processed", r["doc_id"], r["src"])).convert("RGB")
            for r in assets["figure_images"]]
    assert [im.size for im in imgs] == [im.size for im in pages + figs]
    if not dm.CONTRACTS[task].redact_page_numbers:
        assert all(a.tobytes() == b.tobytes() for a, b in zip(imgs, pages + figs))


@needs_fixture
def test_real_caption_prediction_is_the_crop_not_the_page(real):
    s = real["caption_prediction"]
    cooked = cooker_docmaestro(s)
    crop_ref = s["assets.json"]["figure_images"][0]
    crop = Image.open(dm.figure_path(FIXTURE / "processed", crop_ref["doc_id"], crop_ref["src"])).convert("RGB")
    page = Image.open(dm.page_path(FIXTURE / "pages", s["json"]["doc_id"], s["json"]["page_idx"]))
    assert cooked.images is None
    assert cooked.image.tobytes() == crop.tobytes()
    assert cooked.image.size != page.size


@needs_fixture
def test_real_page_sorting_is_redacted(real):
    s = real["page_sorting"]
    imgs = cooker_docmaestro(s).images
    redacted = 0
    for im, ref, entry in zip(imgs, s["assets.json"]["page_images"], s["input.json"]):
        raw = Image.open(dm.page_path(FIXTURE / "pages", ref["doc_id"], ref["page_idx"])).convert("RGB")
        for box in entry["page_number_bboxes"]:
            x1, y1, x2, y2 = dm.parse_bbox(box)
            w, h = im.size
            region = (int(x1 / 1000 * w), int(y1 / 1000 * h), int(x2 / 1000 * w), int(y2 / 1000 * h))
            assert im.crop(region).getextrema() == ((0, 0),) * 3
            assert raw.crop(region).getextrema() != ((0, 0),) * 3   # there was a number there
            redacted += 1
    assert redacted >= 1, "fixture page_sorting sample has no page numbers to redact"


# ---- through the real encoder -------------------------------------------------------

def _processor():
    from transformers import AutoProcessor
    src = os.environ.get("DOCMAESTRO_TEST_PROCESSOR", "Qwen/Qwen3.5-0.8B")
    try:
        proc = AutoProcessor.from_pretrained(src, local_files_only=True, max_pixels=512 * 512)
    except Exception as exc:  # noqa: BLE001 -- any failure means "not available here"
        pytest.skip(f"no processor {src!r} available offline: {type(exc).__name__}")
    template = Path(__file__).resolve().parents[2] / "assets" / "chat_template_sft.jinja"
    proc.chat_template = template.read_text()
    return proc


@pytest.fixture(scope="module")
def encoder():
    from megatron.energon import WorkerConfig
    from data.energon_dataloader import PackedBatchEncoder
    proc = _processor()
    wc = WorkerConfig(rank=0, world_size=1, num_workers=0)
    wc.worker_activate(0)
    merge = proc.image_processor.merge_size
    enc = PackedBatchEncoder(proc, 65536, 65536, image_token_id=proc.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                             video_token_id=-1, spatial_merge_size=merge, max_image_size=512)
    yield enc, merge
    wc.worker_deactivate()


@needs_fixture
@pytest.mark.parametrize("task", TASKS)
def test_real_sample_through_encode_sample(task, real, encoder):
    enc, merge = encoder
    s = real[task]
    n = len(s["assets.json"]["page_images"]) + len(s["assets.json"]["figure_images"])
    packed = enc.encode_sample(cooker_docmaestro(s))
    n_image_tokens = int((packed.input_ids == enc.image_token_id).sum())
    if n == 0:
        assert packed.image_grid_thw is None and n_image_tokens == 0
    else:
        assert packed.image_grid_thw.shape[0] == n
        assert n_image_tokens == int(packed.image_grid_thw.prod(-1).sum()) // (merge * merge)
    assert int((packed.labels != -100).sum()) > 0

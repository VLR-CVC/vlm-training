"""`cooker_text` reads two on-disk shapes, because the two clusters ship
different ones: `.txt` per sample locally, `.json` with a "text" field for
/gpfs/scratch/ehpc543/fineweb_edu on MN5.

CPU-only, no energon dataset needed."""
from data.cookers import cooker_text

DOC = "the quick brown fox"
KEYS = {"__key__": "k", "__restore_key__": (), "__subflavors__": None,
        "__sources__": ()}


def _text(sample):
    cooked = cooker_text(dict(KEYS, **sample))
    assert cooked.messages[-1]["role"] == "assistant", cooked.messages
    return cooked.messages[-1]["content"][0]["text"]


def test_txt_shape():
    assert _text({"txt": DOC}) == DOC


def test_json_shape():
    assert _text({"json": {"text": DOC}}) == DOC


def test_no_vision_payload():
    """image=None is what makes the processor skip pixel_values entirely."""
    assert cooker_text(dict(KEYS, json={"text": DOC})).image is None


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            sys.exit(1)
    print(f"cooker_text: {len(fns)} checks passed")

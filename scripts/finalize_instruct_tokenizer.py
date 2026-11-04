"""Point a converted snapshot's tokenizer at the template the run trained with.

    python scripts/finalize_instruct_tokenizer.py <out-dir> <template> [instruct-ref]

Called by `scripts/convert_final_checkpoint.sh` under QWEN_FINALIZE_INSTRUCT=1;
usable on its own for a snapshot converted by hand.

`dcp_to_hf.py` copies every non-weight file from the BASE snapshot, so a run
finetuned off `*-Base` inherits Base's tokenizer settings. Two of them are wrong
for a chat checkpoint:

  * `eos_token` is `<|endoftext|>` (248044), but the SFT chat template closes
    every assistant turn with `<|im_end|>` (248046), so `generate` never sees a
    stop token and runs to `max_new_tokens`.
  * Base ships no `generation_config.json`, which is where transformers and vLLM
    look first for the stop ids.

The chat template itself lives in two places a loader may read -- the
`chat_template.jinja` file and the `chat_template` key inside
`tokenizer_config.json` -- and the caller has already replaced the file, so this
only has the key left to do.
"""

import json
import shutil
import sys
from pathlib import Path

TURN_END = "<|im_end|>"


def finalize(out: Path, template_path: Path, instruct_ref: str | None = None) -> None:
    template = template_path.read_text()
    cfg_path = out / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())

    # The ids are already in the config; looked up rather than hardcoded so a
    # tokenizer without the turn terminator fails loudly instead of being given a
    # stop token that is not in its vocab.
    ids = {
        entry["content"]: int(tok_id)
        for tok_id, entry in cfg.get("added_tokens_decoder", {}).items()
    }
    if TURN_END not in ids:
        raise SystemExit(
            f"[finalize] {TURN_END} is not in this tokenizer; refusing to set eos_token"
        )

    print(
        f"[finalize]   chat_template {len(cfg.get('chat_template') or '')} -> {len(template)}"
    )
    print(
        f"[finalize]   eos_token {cfg.get('eos_token')} -> {TURN_END}"
        f"  (pad_token {cfg.get('pad_token')} kept)"
    )
    cfg["chat_template"] = template
    cfg["eos_token"] = TURN_END
    cfg_path.write_text(json.dumps(cfg, indent=4, ensure_ascii=False) + "\n")

    gen_path = out / "generation_config.json"
    if instruct_ref:
        src = Path(instruct_ref) / "generation_config.json"
        shutil.copy2(src, gen_path)
        print(f"[finalize]   generation_config.json copied from {src}")
        return

    # Both stop tokens, in the order the official instruct releases use: the turn
    # terminator first, then the document separator a Base-style completion can
    # still emit. No sampling defaults -- those belong to a release, not to us.
    pad = cfg.get("pad_token")
    body: dict = {"eos_token_id": [ids[TURN_END]]}
    if pad in ids:
        body["eos_token_id"].append(ids[pad])
        body["pad_token_id"] = ids[pad]
    gen_path.write_text(json.dumps(body, indent=4) + "\n")
    print(f"[finalize]   generation_config.json written: {body}")


if __name__ == "__main__":
    if not 3 <= len(sys.argv) <= 4:
        raise SystemExit(__doc__)
    ref = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
    finalize(Path(sys.argv[1]), Path(sys.argv[2]), ref)

"""Convert trainer DCP checkpoints -> HF safetensors snapshots.

KEY MAPPING (this is where the old version was silently broken):
  The trainer saves `self.model` (a Qwen3VL *ForConditionalGeneration*) inside a
  state_dict container, and torch.compile wraps submodules, so DCP keys look like:

      model.model.language_model.layers.0.self_attn.q_proj.weight   (double `model.`)
      model.model.visual.merger._orig_mod.norm.weight               (compile `._orig_mod`)
      model.lm_head._orig_mod.weight

  To restore into an HF model we map each DCP key -> HF key by stripping the OUTER
  `model.` container prefix and removing every `._orig_mod`:

      model.model.language_model....  ->  model.language_model....   (HF ForConditionalGeneration)
      model.lm_head._orig_mod.weight  ->  lm_head.weight

  The OLD script used `AutoModel` (the inner Qwen3VLModel, keys `language_model.*`)
  and `target = "model." + key` (single `model.`), and ignored `._orig_mod`. That
  matched 0/625 weights, so `load_state_dict(strict=False)` kept the BASE weights and
  every snapshot was just the untrained base model. We now use
  AutoModelForImageTextToText (-> ForConditionalGeneration, keys `model.<...>` + lm_head)
  and assert a high match rate so a silent miss can never happen again.

Run:
    python utils/convertion_script.py --base_model <hf_dir> --checkpoint_dir <dir_with_checkpoint-step-*>
"""

import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForImageTextToText, AutoProcessor
import argparse
import os
import glob


def dcp_to_hf(dcp_key: str) -> str:
    """Strip the outer trainer `model.` container and any torch.compile `._orig_mod`."""
    key = dcp_key[len("model."):] if dcp_key.startswith("model.") else dcp_key
    return key.replace("._orig_mod", "")


def convert_nested_dcp_batch(base_model_path, checkpoint_dir, min_match_ratio=0.95):
    models_out_dir = os.path.join(checkpoint_dir, "models")
    os.makedirs(models_out_dir, exist_ok=True)

    search_pattern = os.path.join(checkpoint_dir, "checkpoint-step-*")
    checkpoint_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    if not checkpoint_dirs:
        print(f"No checkpoints found in {checkpoint_dir} matching 'checkpoint-step-*'")
        return

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)

    ref_state = model.state_dict()          # HF ForConditionalGeneration keys
    ref_keys = set(ref_state.keys())

    for ckpt_path in sorted(checkpoint_dirs):
        step_name = os.path.basename(ckpt_path)
        step_num = step_name.split("-")[-1]
        output_path = os.path.join(models_out_dir, f"step-{step_num}")
        print(f"Processing {step_name} -> {output_path}")

        reader = dcp.FileSystemReader(ckpt_path)
        checkpoint_keys = set(reader.read_metadata().state_dict_metadata.keys())

        # Build the load plan keyed by the DCP key, with correctly-shaped CPU buffers
        # taken from the reference HF model. dcp_key -> hf_key.
        dcp_to_hf_map = {}
        load_plan = {}
        for dk in checkpoint_keys:
            hk = dcp_to_hf(dk)
            if hk in ref_keys:
                dcp_to_hf_map[dk] = hk
                load_plan[dk] = torch.empty_like(ref_state[hk], device="cpu")

        n_match, n_ref = len(load_plan), len(ref_keys)
        # lm_head is tied -> absent from the checkpoint; don't count it against us.
        tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        effective_ref = n_ref - (1 if tied and "lm_head.weight" in ref_keys else 0)
        ratio = n_match / max(1, effective_ref)
        print(f"  matched {n_match}/{n_ref} HF tensors from checkpoint (ratio={ratio:.3f})")
        if ratio < min_match_ratio:
            raise RuntimeError(
                f"Only {n_match}/{effective_ref} weights matched (ratio {ratio:.3f} < "
                f"{min_match_ratio}). Key mapping is wrong -- refusing to write a snapshot "
                f"that would silently be the untrained base model. "
                f"Example checkpoint keys: {sorted(checkpoint_keys)[:3]}"
            )

        dcp.load(state_dict=load_plan, checkpoint_id=ckpt_path)

        restored_state_dict = {dcp_to_hf_map[dk]: t for dk, t in load_plan.items()}
        missing, unexpected = model.load_state_dict(restored_state_dict, strict=False)
        missing = [m for m in missing if not (tied and m == "lm_head.weight")]
        if missing:
            print(f"  WARNING: {len(missing)} weights NOT restored (kept base), e.g. {missing[:5]}")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")

        processor.save_pretrained(output_path)
        model.save_pretrained(output_path, safe_serialization=True)
        print(f"Saved HF snapshot to {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to original HF model")
    parser.add_argument("--checkpoint_dir", required=True,
                        help="Path to the directory containing checkpoint folders")
    parser.add_argument("--min_match_ratio", type=float, default=0.95,
                        help="Fail if fewer than this fraction of model weights are restored")
    args = parser.parse_args()
    convert_nested_dcp_batch(args.base_model, args.checkpoint_dir, args.min_match_ratio)

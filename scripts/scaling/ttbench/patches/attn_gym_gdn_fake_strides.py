#!/usr/bin/env python3
"""Fix attn_gym's GDN fake kernels, which lie about their output layouts.

`attn_gym/linear/gdn/ops.py:185` allocates the fake output with

    torch.empty_like(v, dtype=q.dtype)

and `empty_like` inherits v's strides. When v arrives permuted, the fake
advertises a permuted output -- but the real Triton kernel
(`chunk_gdn_fwd_output_dense`, impl/chunk.py:108) returns a fresh contiguous
tensor. Inductor plans downstream ops against the advertised layout, and its
guard fires as soon as a graph carrying that assumption is reached:

    assert_tensor_metadata(buf32, (1, 10240, 8, 128),
                           (10485760, 1, 1310720, 10240), torch.bfloat16,
                           'torch.ops.attn_gym.gdn_chunk_fwd.default')
    AssertionError: expected size 10240==10240, stride 1024==1 at dim=1; ...
    Error in op: torch.ops.attn_gym.gdn_chunk_fwd.default

Killed TORCHTITAN_BENCHMARK.md phase 4 at 4n step 38 (job 1811111), 16n step 15
(1811112), 32n step 37 (1811114) and the static-shape retry at step 38 (1811333).
Not a dynamic-shape problem: 1811333 ran with `automatic_dynamic_shapes = False`
and produced the concrete guard quoted above.

`TORCHINDUCTOR_SIZE_ASSERTS=0` is NOT the fix. The assert compares what inductor
expected against what the op returned; muting it hands the mis-strided tensor to
kernels compiled for the other layout, trading a crash for a plausible wrong
number.

Every fake in this file allocates its outputs with `empty_like`, and every real
impl allocates fresh contiguous tensors, so all of them carry the same defect.
Patching only the forward (first version of this script) simply moved the failure
one call deeper, to `gdn_chunk_bwd` at 4n step 38 and 16n step 15 (jobs 1811583,
1811584) -- the backward's five grad outputs are the same `empty_like` pattern.
So the rule here is uniform: a GDN fake output is contiguous.

Idempotent. Run against the benchmark venv only -- never torch_main:

    python attn_gym_gdn_fake_strides.py \
        $TT_ROOT/venv/lib/python3.13/site-packages/attn_gym/linear/gdn/ops.py
"""

import pathlib
import sys

# Every GDN fake allocates outputs with empty_like, inheriting whatever strides
# the input happened to have. Each (old, new) pair below makes the claimed layout
# contiguous, matching what the Triton kernels actually return.
REPLACEMENTS = [
    # _chunk_fwd_fake, _chunk_fwd_packed_fake, _recurrent_fwd_fake and friends:
    # the output is a fresh tensor shaped like v, in q's dtype.
    (
        "return torch.empty_like(v, dtype=q.dtype)",
        "return torch.empty(v.shape, dtype=q.dtype, device=v.device)",
    ),
    # _chunk_bwd_fake: five gradients, one per differentiable input.
    (
        """    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(cumulative_gate),
        torch.empty_like(beta),""",
        """    # PATCHED (qwen4 TORCHTITAN_BENCHMARK.md): see module docstring --
    # empty_like inherits input strides, chunk_gdn_bwd returns contiguous.
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(k.shape, dtype=k.dtype, device=k.device),
        torch.empty(v.shape, dtype=v.dtype, device=v.device),
        torch.empty(cumulative_gate.shape, dtype=cumulative_gate.dtype,
                    device=cumulative_gate.device),
        torch.empty(beta.shape, dtype=beta.dtype, device=beta.device),""",
    ),
]


def main() -> int:
    path = pathlib.Path(sys.argv[1])
    source = path.read_text()
    original = source

    total = 0
    for old, new in REPLACEMENTS:
        count = source.count(old)
        total += count
        if count:
            source = source.replace(old, new)

    if source == original:
        if "PATCHED (qwen4" in original:
            print(f"already patched: {path}")
            return 0
        print("no matches -- attn_gym version drift?")
        return 1

    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        backup.write_text(original)
        print(f"backup: {backup}")

    path.write_text(source)
    print(f"patched {total} site(s): {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

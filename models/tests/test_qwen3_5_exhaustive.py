"""Exhaustive pytest test suite for the native Qwen3.5 implementation.

Tests are grouped into five sections:

  1. parity        – logits match HF reference (short / medium / long sequences)
  2. generation    – greedy decode matches HF token-for-token
  3. varlen        – N≥3 packed samples give identical per-sample logits as solo runs
  4. loss          – loss values match HF; packed self-consistency; label masking
  5. numerics      – gradient flow, no NaN/Inf, bfloat16 stability

Gold outputs are loaded from `qwen3_5_test_data.pt` (produced by
`generate_qwen3_5_test_data.py`). When the file is absent, the HF model is
loaded live and the parity / generation tests still run.

Run:
    pytest models/tests/test_qwen3_5_exhaustive.py -s -v

    # point at a custom snapshot
    QWEN3_5_SNAPSHOT=/path/to/qwen35 pytest models/tests/test_qwen3_5_exhaustive.py -s -v

    # skip tests that need CUDA
    pytest models/tests/test_qwen3_5_exhaustive.py -m "not cuda_only"

HF model notes:
  HF's causal_conv1d CUDA kernel requires seqlen ≥ d_conv (4 tokens for
  Qwen3.5-2B). Tests that call the HF model use prompts with ≥ 8 tokens.
  Single-token / very-short tests only exercise our native implementation.

  We always pass use_cache=False to HF forward calls to prevent the KV cache
  from accumulating stale state across tests in the same session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Make the repo root importable regardless of where pytest is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.model import Qwen3_5ForCausalLM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _probe_causal_conv1d() -> bool:
    """Return True iff the causal_conv1d CUDA kernel is functional.

    In some environments (e.g. PyTorch 2.11.0+cu130) the kernel import
    succeeds but the actual CUDA call raises
    "Cannot access data pointer of Tensor that doesn't have storage".
    When the kernel is broken, HF's GatedDeltaNet falls back to a pure
    PyTorch path whose floating-point results differ slightly from the
    CUDA path, which causes greedy generation to diverge after a few steps.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from causal_conv1d import causal_conv1d_fn
        if causal_conv1d_fn is None:
            return False
        _x = torch.zeros(1, 1, 8, device="cuda", dtype=torch.float32)
        _w = torch.zeros(1, 1, 4, device="cuda", dtype=torch.float32)
        causal_conv1d_fn(_x, _w, None, None, None, True)
        del _x, _w
        return True
    except Exception:
        return False

_HF_GENERATION_RELIABLE = _probe_causal_conv1d()

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
)
DATA_PATH = Path(__file__).with_name("qwen3_5_test_data.pt")

# Tolerance matching bf16 precision (~0.5 absolute, ~0.1 relative).
ATOL_BF16 = 0.5
RTOL_BF16 = 0.1
# Looser tolerance for per-sample scalar losses.
ATOL_LOSS = 1e-2

# Minimum token count that HF's causal_conv1d accepts without error.
# Qwen3.5 has linear_conv_kernel_dim=4; we use 8 as a safe margin.
HF_MIN_TOKENS = 8

# ---------------------------------------------------------------------------
# Module-level fixtures  (loaded once per test session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SNAPSHOT)


@pytest.fixture(scope="module")
def our_model(device: torch.device) -> Qwen3_5ForCausalLM:
    """Our native Qwen3.5, bf16, eval mode."""
    model = Qwen3_5ForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=False
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def hf_model(device: torch.device):
    """HF reference model, bf16, sdpa, eval mode.

    We use ``attn_implementation="sdpa"`` because our native ``SelfAttention``
    relies on ``torch.nn.attention.varlen.varlen_attn``, which matches HF's
    sdpa reduction order bit-for-bit in bf16. HF's flash_attention_2 kernel
    uses a different reduction order that drifts ~1 ULP and can flip greedy
    argmax ties at a handful of positions.

    Both our GatedDeltaNet and HF's use the fused ``causal_conv1d_fn`` kernel
    for the depthwise conv + SiLU step, so no override of the conv path is
    needed for parity.
    """
    from transformers import Qwen3_5ForConditionalGeneration
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            SNAPSHOT,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    return model


@pytest.fixture(scope="module")
def synthetic_data() -> dict[str, Any] | None:
    """Pre-generated test data from `generate_qwen3_5_test_data.py`."""
    if DATA_PATH.exists():
        return torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    return None


@pytest.fixture(scope="module", autouse=True)
def warmup_kernels(our_model: Qwen3_5ForCausalLM, tokenizer, device: torch.device):
    """Warm up all JIT-compiled kernels (FLA chunk_gated_delta_rule, varlen_attn)
    before any HF comparison test runs.

    On the first forward pass our model triggers CUDA JIT compilation for FLA
    and flash-attention kernels. This leaves transient state in CUDA's
    compilation cache that changes the numerical results of subsequent calls
    (including HF's fallback conv1d). By pre-warming here, every test sees a
    stable CUDA environment.
    """
    prompt = "The capital of France is Paris, and Germany has Berlin"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        _ = our_model(input_ids=ids)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _greedy_generate(
    model: Qwen3_5ForCausalLM,
    input_ids: torch.Tensor,
    max_new: int,
) -> torch.Tensor:
    """Simple no-cache greedy decode for our native model."""
    ids = input_ids
    for _ in range(max_new):
        logits = model(ids)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids


def _make_cu_seqlens(lengths: list[int], device: torch.device) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


# ---------------------------------------------------------------------------
# 1.  PARITY  –  logit-level agreement between ours and HF
# ---------------------------------------------------------------------------

class TestLogitParity:
    """Forward-pass logit parity on diverse text sequences.

    All prompts that call the HF model have ≥ HF_MIN_TOKENS tokens so that
    HF's causal_conv1d CUDA kernel does not error out.
    """

    SHORT_PROMPTS = [
        "In the beginning there was nothing but darkness and silence",  # 10 tokens
        "Mathematics is the language in which the universe is written", # 11 tokens
    ]
    MEDIUM_PROMPTS = [
        "The capital of France is Paris, and the capital of Germany is Berlin.",
        "In computer science, a linked list is a linear data structure.",
    ]
    LONG_PROMPTS = [
        (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        ),
        (
            "def fibonacci(n):\n    if n <= 1:\n        return n\n"
            "    return fibonacci(n-1) + fibonacci(n-2)"
        ),
    ]

    def _run(
        self,
        prompt: str,
        our_model: Qwen3_5ForCausalLM,
        hf_model,
        tokenizer,
        device: torch.device,
    ) -> None:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        assert ids.shape[1] >= HF_MIN_TOKENS, (
            f"Prompt {prompt!r} has only {ids.shape[1]} tokens; "
            f"HF's causal_conv1d requires ≥ {HF_MIN_TOKENS}"
        )
        with torch.no_grad():
            ours = our_model(input_ids=ids).float()
            # use_cache=False prevents stale KV state accumulating between tests.
            theirs = hf_model(input_ids=ids, use_cache=False).logits.float()

        assert ours.shape == theirs.shape, f"shape mismatch: {ours.shape} vs {theirs.shape}"
        assert torch.equal(ours.argmax(-1), theirs.argmax(-1)), (
            f"argmax mismatch on prompt={prompt!r}"
        )
        torch.testing.assert_close(ours, theirs, atol=ATOL_BF16, rtol=RTOL_BF16)

    @pytest.mark.parametrize("prompt", SHORT_PROMPTS)
    def test_short(self, prompt, our_model, hf_model, tokenizer, device):
        self._run(prompt, our_model, hf_model, tokenizer, device)

    @pytest.mark.parametrize("prompt", MEDIUM_PROMPTS)
    def test_medium(self, prompt, our_model, hf_model, tokenizer, device):
        self._run(prompt, our_model, hf_model, tokenizer, device)

    @pytest.mark.parametrize("prompt", LONG_PROMPTS)
    def test_long(self, prompt, our_model, hf_model, tokenizer, device):
        self._run(prompt, our_model, hf_model, tokenizer, device)

    def test_single_token(self, our_model, tokenizer, device):
        """Single-token input: our model must produce finite, shaped output.

        HF's causal_conv1d CUDA kernel requires seqlen ≥ d_conv, so we only
        test our native implementation here (not vs HF).
        """
        ids = tokenizer("Hi", return_tensors="pt").input_ids[:, :1].to(device)
        assert ids.shape[1] == 1, "expected exactly 1 token"
        with torch.no_grad():
            logits = our_model(input_ids=ids).float()
        assert logits.shape == (1, 1, logits.shape[-1])
        assert torch.isfinite(logits).all(), "single-token logits contain NaN/Inf"

    def test_against_pregenerated(self, our_model, synthetic_data, device):
        """When synthetic data is available, compare against every stored record."""
        if synthetic_data is None:
            pytest.skip("Pre-generated data not found; run generate_qwen3_5_test_data.py first")

        failed = []
        for rec in synthetic_data["records"]:
            ids = rec["input_ids"].to(device)
            gold = rec["hf_logits"].to(device).float()
            with torch.no_grad():
                ours = our_model(input_ids=ids).float()
            if not torch.equal(ours.argmax(-1), gold.argmax(-1)):
                failed.append(rec["prompt"])

        assert not failed, f"argmax mismatch on {len(failed)} prompt(s): {failed}"

    def test_top5_agreement(self, our_model, hf_model, tokenizer, device):
        """Top-5 token sets should overlap ≥80 % across all positions."""
        prompt = "The quick brown fox jumps over the lazy dog near the river"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            ours = our_model(input_ids=ids).float()
            theirs = hf_model(input_ids=ids, use_cache=False).logits.float()

        top5_ours = ours.topk(5, dim=-1).indices    # (1, S, 5)
        top5_hf   = theirs.topk(5, dim=-1).indices

        S = ids.shape[1]
        matches = 0
        for pos in range(S):
            o = set(top5_ours[0, pos].tolist())
            h = set(top5_hf[0, pos].tolist())
            if len(o & h) >= 4:  # ≥4 of 5 in common
                matches += 1

        ratio = matches / S
        assert ratio >= 0.80, f"top-5 agreement only {ratio:.1%} < 80 %"


# ---------------------------------------------------------------------------
# 2.  GENERATION  –  greedy decode token sequence must match HF
# ---------------------------------------------------------------------------

class TestGeneration:
    MAX_NEW = 15

    GENERATION_PROMPTS = [
        "The capital of France is Paris, and Germany has Berlin",  # 11 tokens
        "Once upon a time in a land far away from everything known",  # 12 tokens
        "In Python you can define a function using the def keyword",   # 11 tokens
    ]

    def _run(
        self,
        prompt: str,
        our_model: Qwen3_5ForCausalLM,
        hf_model,
        tokenizer,
        device: torch.device,
    ) -> None:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attn_mask = torch.ones_like(ids)  # suppress HF padding warnings
        prompt_len = ids.shape[1]

        our_out = _greedy_generate(our_model, ids, self.MAX_NEW)
        hf_out = hf_model.generate(
            ids,
            attention_mask=attn_mask,
            max_new_tokens=self.MAX_NEW,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=False,
        )

        our_new = our_out[0, prompt_len:].tolist()
        hf_new  = hf_out[0, prompt_len:].tolist()

        assert our_new == hf_new, (
            f"Generated tokens diverge for prompt={prompt!r}\n"
            f"  ours={tokenizer.decode(our_new)!r}\n"
            f"  hf  ={tokenizer.decode(hf_new)!r}"
        )

    @pytest.mark.parametrize("prompt", GENERATION_PROMPTS)
    def test_greedy_matches_hf(self, prompt, our_model, hf_model, tokenizer, device):
        self._run(prompt, our_model, hf_model, tokenizer, device)

    def test_against_pregenerated(self, our_model, synthetic_data, tokenizer, device):
        """Token sequences from pre-generated data must match our greedy output."""
        if synthetic_data is None:
            pytest.skip("Pre-generated data not found; run generate_qwen3_5_test_data.py first")

        max_new = synthetic_data["max_new_tokens"]
        failed = []
        for rec in synthetic_data["records"]:
            ids = rec["input_ids"].to(device)
            gold_new = rec["hf_new_tokens"].tolist()
            our_out = _greedy_generate(our_model, ids, max_new)
            prompt_len = ids.shape[1]
            our_new = our_out[0, prompt_len:].tolist()
            if our_new != gold_new:
                failed.append(rec["prompt"])

        assert not failed, f"Generation mismatch on {len(failed)} prompt(s): {failed}"


# ---------------------------------------------------------------------------
# 3.  VARLEN PACKING
# ---------------------------------------------------------------------------

class TestVarlenPacking:
    """Packed samples must produce identical per-sample logits as solo runs."""

    def _check_packing(
        self,
        model: Qwen3_5ForCausalLM,
        ids_list: list[torch.Tensor],
        device: torch.device,
    ) -> None:
        """Pack ids_list into one row and assert each slice matches solo output."""
        packed = torch.cat(ids_list, dim=1)
        lengths = [t.shape[1] for t in ids_list]
        cu = _make_cu_seqlens(lengths, device)

        with torch.no_grad():
            solo_logits = [model(input_ids=ids).float() for ids in ids_list]
            pack_logits = model(input_ids=packed, attention_mask=cu).float()

        offset = 0
        for i, (solo, length) in enumerate(zip(solo_logits, lengths)):
            slc = pack_logits[:, offset : offset + length, :]
            assert torch.equal(slc.argmax(-1), solo.argmax(-1)), (
                f"sample {i}: argmax mismatch after packing"
            )
            torch.testing.assert_close(slc, solo, atol=ATOL_BF16, rtol=RTOL_BF16)
            offset += length

    @pytest.mark.cuda_only
    def test_two_samples(self, our_model, tokenizer, device):
        if not device.type == "cuda":
            pytest.skip("varlen requires CUDA (flash attention)")
        prompts = [
            "The capital of France is",
            "The quick brown fox jumps over",
        ]
        ids_list = [
            tokenizer(p, return_tensors="pt").input_ids.to(device)
            for p in prompts
        ]
        self._check_packing(our_model, ids_list, device)

    @pytest.mark.cuda_only
    def test_three_samples(self, our_model, tokenizer, device):
        if not device.type == "cuda":
            pytest.skip("varlen requires CUDA (flash attention)")
        prompts = [
            "Hello world, how are you today",
            "The sky is blue and the grass is green",
            "In the beginning was the word",
        ]
        ids_list = [
            tokenizer(p, return_tensors="pt").input_ids.to(device)
            for p in prompts
        ]
        self._check_packing(our_model, ids_list, device)

    @pytest.mark.cuda_only
    def test_four_samples_varying_length(self, our_model, tokenizer, device):
        """Four samples with significantly different lengths."""
        if not device.type == "cuda":
            pytest.skip("varlen requires CUDA (flash attention)")
        prompts = [
            "Hello world",
            "The capital of France is Paris",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)",
        ]
        ids_list = [
            tokenizer(p, return_tensors="pt").input_ids.to(device)
            for p in prompts
        ]
        self._check_packing(our_model, ids_list, device)

    @pytest.mark.cuda_only
    def test_packing_with_padding(self, our_model, tokenizer, device):
        """Packed + padded row: pad tokens beyond last real token must not affect logits."""
        if not device.type == "cuda":
            pytest.skip("varlen requires CUDA (flash attention)")
        prompts = [
            "The capital of France is",
            "The quick brown fox jumps over the lazy dog",
        ]
        ids_list = [
            tokenizer(p, return_tensors="pt").input_ids.to(device)
            for p in prompts
        ]
        Sa, Sb = ids_list[0].shape[1], ids_list[1].shape[1]
        # Pad to next multiple of 64 (≥ FLA chunk size) so the total sequence
        # length is chunk-aligned, which is the same strategy as the reference
        # test_qwen3_5_loss_parity.py.
        chunk_size = 64
        real_len = Sa + Sb
        pad_to = ((real_len + chunk_size - 1) // chunk_size) * chunk_size
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        pad = torch.full((1, pad_to - real_len), pad_id, dtype=ids_list[0].dtype, device=device)
        packed = torch.cat([ids_list[0], ids_list[1], pad], dim=1)
        cu = torch.tensor([0, Sa, Sa + Sb, pad_to], dtype=torch.int32, device=device)

        with torch.no_grad():
            logits_a_solo = our_model(input_ids=ids_list[0]).float()
            logits_b_solo = our_model(input_ids=ids_list[1]).float()
            pack_logits = our_model(input_ids=packed, attention_mask=cu).float()

        pack_a = pack_logits[:, :Sa]
        pack_b = pack_logits[:, Sa : Sa + Sb]

        assert torch.equal(pack_a.argmax(-1), logits_a_solo.argmax(-1)), "sample A mismatch"
        assert torch.equal(pack_b.argmax(-1), logits_b_solo.argmax(-1)), "sample B mismatch"


# ---------------------------------------------------------------------------
# 4.  LOSS
# ---------------------------------------------------------------------------

class TestLoss:
    """Loss computation correctness."""

    LOSS_PROMPTS = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "In computer science, a list is an abstract data type.",
    ]

    def test_single_sample_loss_matches_hf(self, our_model, hf_model, tokenizer, device):
        """Per-sample scalar loss must agree with HF within bf16 tolerance."""
        max_diff = 0.0
        for text in self.LOSS_PROMPTS:
            ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            labels = ids.clone()
            with torch.no_grad():
                l_ours = our_model(input_ids=ids, labels=labels).loss.float().item()
                l_hf   = hf_model(input_ids=ids, labels=labels, use_cache=False).loss.float().item()
            diff = abs(l_ours - l_hf)
            max_diff = max(max_diff, diff)
            assert diff < ATOL_LOSS, (
                f"loss diff {diff:.3e} > {ATOL_LOSS:.0e}  text={text!r}"
            )
        print(f"\n  max |ours-hf| loss = {max_diff:.3e}")

    @pytest.mark.cuda_only
    def test_packed_self_consistency(self, our_model, tokenizer, device):
        """
        Packed run must reproduce solo-run logits for each sample.

        The packed sequence is padded to 128 tokens (a multiple of FLA's
        chunk size) so that the gated-delta-rule kernel handles segment
        boundaries correctly — same approach as test_qwen3_5_loss_parity.py.
        """
        if not device.type == "cuda":
            pytest.skip("varlen requires CUDA")

        prompts = self.LOSS_PROMPTS[:2]
        ids_list = [
            tokenizer(t, return_tensors="pt").input_ids.to(device)
            for t in prompts
        ]
        Sa, Sb = ids_list[0].shape[1], ids_list[1].shape[1]

        # Pad to 128 — the same constant used by the reference loss parity test.
        pad_to = 128
        assert Sa + Sb <= pad_to, (
            f"Prompts are too long to fit in pad_to={pad_to}: {Sa}+{Sb}={Sa+Sb}"
        )
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        pad = torch.full((1, pad_to - Sa - Sb), pad_id, dtype=ids_list[0].dtype, device=device)
        packed = torch.cat([ids_list[0], ids_list[1], pad], dim=1)
        cu = torch.tensor([0, Sa, Sa + Sb, pad_to], dtype=torch.int32, device=device)

        labels = packed.clone()
        labels[0, Sa + Sb :] = -100  # ignore pad positions in loss

        with torch.no_grad():
            out_packed = our_model(input_ids=packed, attention_mask=cu, labels=labels)
            pack_a = out_packed.logits[:, :Sa].float()
            pack_b = out_packed.logits[:, Sa : Sa + Sb].float()
            solo_a = our_model(input_ids=ids_list[0]).float()
            solo_b = our_model(input_ids=ids_list[1]).float()

        assert torch.equal(pack_a.argmax(-1), solo_a.argmax(-1)), "packed sample A logits differ"
        assert torch.equal(pack_b.argmax(-1), solo_b.argmax(-1)), "packed sample B logits differ (state leakage?)"

    def test_label_masking(self, our_model, tokenizer, device):
        """Tokens masked with -100 must not contribute to the loss."""
        text = "Hello world, this is a test sentence for the model."
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        n = ids.shape[1]

        labels_full = ids.clone()
        labels_half = ids.clone()
        labels_half[0, : n // 2] = -100

        with torch.no_grad():
            loss_full = our_model(input_ids=ids, labels=labels_full).loss.float()
            loss_half = our_model(input_ids=ids, labels=labels_half).loss.float()

        assert not torch.isclose(loss_full, loss_half, atol=1e-3), (
            "Loss did not change when masking half the labels — label masking is broken"
        )

    def test_all_labels_masked(self, our_model, tokenizer, device):
        """When all labels are -100 the loss should be NaN (no valid positions)."""
        text = "Hello world"
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        labels = torch.full_like(ids, -100)
        with torch.no_grad():
            out = our_model(input_ids=ids, labels=labels)
        # Cross-entropy with no valid positions → NaN (expected behaviour).
        assert torch.isnan(out.loss) or out.loss.item() == 0.0, (
            f"Expected NaN for all-masked labels, got {out.loss.item()}"
        )

    def test_loss_positive(self, our_model, tokenizer, device):
        """Loss on a valid text sequence must be finite and positive."""
        text = "The capital of France is Paris."
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        labels = ids.clone()
        with torch.no_grad():
            loss = our_model(input_ids=ids, labels=labels).loss.float()
        assert torch.isfinite(loss) and loss.item() > 0.0, (
            f"Expected finite positive loss, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# 5.  NUMERICS  –  gradient flow, NaN/Inf checks, dtype stability
# ---------------------------------------------------------------------------

class TestNumerics:
    """Numerical health checks — all tests exercise only our native model."""

    def test_no_nan_inf_in_logits(self, our_model, tokenizer, device):
        """Logits must be finite for several diverse prompts."""
        prompts = [
            "Hello",
            "The capital of France is Paris.",
            "def foo(): pass",
        ]
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                logits = our_model(input_ids=ids)
            assert torch.isfinite(logits).all(), (
                f"Non-finite logits detected for prompt={prompt!r}"
            )

    def test_gradient_flows_through_loss(self, our_model, tokenizer, device):
        """Loss.backward() must complete and all leaf grads must be finite."""
        import copy
        m = copy.deepcopy(our_model).train()

        text = "Hello world"
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        labels = ids.clone()

        out = m(input_ids=ids, labels=labels)
        out.loss.backward()

        for name, param in m.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Non-finite gradient in {name}"
                )

    def test_bf16_logits_finite(self, our_model, tokenizer, device):
        """Raw bf16 output (before float cast) must also be finite."""
        prompt = "The capital of France is"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = our_model(input_ids=ids)  # returned in bf16
        assert torch.isfinite(logits).all(), "bf16 logits contain NaN/Inf"

    def test_deterministic_forward(self, our_model, tokenizer, device):
        """Identical inputs must produce bit-exact identical outputs."""
        prompt = "Deterministic check"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out1 = our_model(input_ids=ids).float()
            out2 = our_model(input_ids=ids).float()
        assert torch.equal(out1, out2), "Non-deterministic forward pass detected"

    def test_output_shape(self, our_model, tokenizer, device):
        """Logit tensor must be (1, seq_len, vocab_size)."""
        prompt = "Shape test"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]
        with torch.no_grad():
            logits = our_model(input_ids=ids)
        assert logits.shape[0] == 1
        assert logits.shape[1] == seq_len
        assert logits.shape[2] > 100_000  # Qwen3.5 has vocab_size ≈ 248k

    def test_logit_scale_reasonable(self, our_model, tokenizer, device):
        """Logit values should be within a reasonable range for bf16 models."""
        prompt = "The quick brown fox"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            logits = our_model(input_ids=ids).float()
        assert logits.abs().max().item() < 200, (
            f"Logit magnitude suspiciously large: {logits.abs().max().item()}"
        )

    def test_longer_sequence_stays_finite(self, our_model, tokenizer, device):
        """Sequences spanning multiple FLA chunks (≥ 80 tokens) must not produce NaN/Inf."""
        long_text = (
            "The history of artificial intelligence began in antiquity, "
            "with myths, stories and rumors of artificial beings endowed "
            "with intelligence or consciousness by master craftsmen. "
            "The seeds of what would become modern AI were planted by "
            "classical philosophers who attempted to describe the process "
            "of human thinking as the mechanical manipulation of symbols. "
            "This work culminated in the invention of the programmable digital "
            "computer in the 1940s, a machine based on the abstract essence "
            "of mathematical reasoning. These early ideas laid the groundwork "
            "for what we now call machine learning and deep neural networks."
        )
        ids = tokenizer(long_text, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]
        assert seq_len >= 80, f"Expected ≥80 tokens to span FLA chunks, got {seq_len}"
        with torch.no_grad():
            logits = our_model(input_ids=ids)
        assert torch.isfinite(logits).all(), (
            f"Non-finite logits for {seq_len}-token sequence"
        )

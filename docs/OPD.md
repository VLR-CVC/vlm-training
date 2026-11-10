# Plan: on-policy distillation (OPD) as a training method

Status: M0 implemented (`train/train_opd.py`, `rollout = "dataset"` only; the
`[opd]` section has the teacher and loss fields, the rollout fields arrive with M1).
M1 onwards is plan. Revises the knowledge-distillation plan of 5444566.
OPD becomes a training method of its own, next to SFT and pretraining, with its own
entry point and config section. Off-policy distillation (KD) is the same method with
the completions read from the dataset instead of sampled, so it comes for free and is
the first milestone.

## What the method does

Each optimizer step:

1. Each DP rank reads a batch of prompts.
2. The student samples completions with its current weights (exactly on-policy).
3. Prompt + completion rows are packed to `seq_len`, as SFT rows are.
4. A frozen teacher runs a no-grad forward on the same rows.
5. The student runs a forward, and the loss is a per-token divergence between the two
   distributions over the full vocabulary, on the completion tokens.
6. Backward, clip, optimizer step: unchanged from SFT.

This is GKD (Agarwal et al. 2024) and the Thinking Machines on-policy distillation
recipe (Lu et al. 2025), with an exact full-vocabulary reverse KL in place of their
sampled-token estimate. There is no reward, no advantage, no value model, no
importance ratio and no clipping. The teacher is local, so the full distribution is
available at every position.

The teacher must share the student's tokenizer and vocabulary.

| use | teacher | student |
|---|---|---|
| post-training after an architecture change (sparse attention, attention-kind swaps, context extension) | the original checkpoint | same weights, changed architecture |
| capability-preserving domain fine-tune | the starting checkpoint | the fine-tuned model, plus `ce_weight` on the domain data |
| compression | Qwen3.5-9B / 35B-A3B | Qwen3.5-2B / 0.8B |

## Decision: a new entry point in this trainer

Not an RL framework. verl, slime, NeMo-RL, prime-rl and TRL would need our models
ported to their trainer and their rollout engine:

- Qwen3.5 with GatedDeltaNet;
- the vision tower;
- the MoE with EP;
- the DSA student.

We would also lose the long-context path:

- packing;
- selective AC;
- chunked loss;
- FSDP2/EP;
- SR AdamW.

The pinned torchtitan (b21f7d43e) ships TitanRL (`torchtitan/experiments/rl`). It is
the closest fit, because it runs the torchtitan model definitions, GDN hybrids
included, inside vLLM. But it is text-only, has no distillation loss, and needs a
vLLM nightly, a PyTorch nightly, Monarch and TorchStore. It stays the reference for
the decode kernels and for a later fast-rollout engine (M4); we do not adopt it.

The layout:

- `train/train_opd.py`: `OPDTrainer(Trainer)` and `OPDConfig(Config)`, which adds
  the `[opd]` section. It is run as `python -m train.train_opd`.
  - It overrides `batch_generator`, which runs prompts, then rollouts, then packed
    rows.
  - It overrides `__init__`, to set up the teacher.
  - The optimizer, scheduler, checkpointing, resume, precompile, profiling and
    logging are inherited.
- `train/step.py` and `train/loss.py` get a `teacher=None` path. With `None`, SFT is
  unchanged.
- `train/rollout.py`: prompt batching, the lockstep sampling loop and packing back
  into rows.
- Model code: additive decode methods only (see the decode section). The training
  forward is untouched.
- Launchers: `scripts/multinode_{jup,mn5}.sh` read `TRAIN_MODULE`, default
  `train.train_qwen`.

SFT tomls stay valid: `[opd]` exists only in `OPDConfig`. There is no copy of
`train_qwen.py`.

## Config: `[opd]`

```toml
[opd]
teacher_dir = ""            # HF snapshot; = training.model_dir for self-distillation
teacher_model_config = ""   # empty: teacher_dir/config.json
teacher_attn_backend = ""   # empty: model.attn_backend
rollout = "student"         # student: sample (OPD); dataset: the data's answers (KD)
divergence = "rkl"          # rkl: KL(student || teacher); fkl: KL(teacher || student)
kl_weight = 1.0
ce_weight = 0.0             # next-token loss on dataset answers; only with rollout = "dataset"
prompt_kl_weight = 0.0      # teacher-forced KL on prompt tokens too
prompts_per_rank = 32       # per optimizer step
samples_per_prompt = 1
max_prompt_len = 16384      # longer prompts are skipped and counted
max_new_tokens = 4096
temperature = 1.0           # 1.0 keeps the samples on-policy
top_p = 1.0
```

- `rkl` is the default for `student` rollouts: it is mode-seeking, and it is what
  OPD uses.
- `fkl` is the usual choice for `dataset` rollouts.
- Startup checks:
  - teacher vocab == student vocab, read from `vocab_size` or
    `text_config.vocab_size`;
  - `loss_chunks > 1`;
  - `tp_size = 1` (see Limits);
  - `ce_weight > 0` only with `rollout = "dataset"`.

## Components

### Prompts

The prompt source is the energon pipeline in a prompt-only mode:

- the cooker's messages, cut before the last assistant turn;
- `apply_chat_template(..., add_generation_prompt=True)`; SFT encoding uses `False`
  (`data/energon_dataloader.py`).

With `rollout = "dataset"`, the cut-off answer is kept as the completion, which is
the current SFT encoding. The prompt-only mode is therefore a flag on the task
encoder, not a second loader.

Image prompts keep their `pixel_values` and `grid_thw`.

### Rollout: native decode in our models

Sampling runs on the training GPUs with the training module. There is no weight sync
and no second copy of the model, the samples are exactly on-policy, and it works for
every architecture we train. The DSA student and `qwen3_5_text` attention swaps are
the reason the off-the-shelf engines do not fit.

Per rollout:

1. `model.unshard()` once: FSDP all-gathers per decoded token would dominate.
   Weights resident for the rollout:
   - 2B: 4.6 GB;
   - 9B: 18 GB.

   The memory freed after the previous backward covers this.
2. **Prefill**: the existing varlen forward on the packed prompts, returning the
   caches:
   - K/V of the full-attention layers;
   - GDN recurrent state, from `chunk_gdn` with the final state on;
   - the last `conv_kernel - 1` inputs of each GDN causal conv.

   The vision tower runs only here.
3. **Decode loop**: one token for every live sequence per step, using:
   - flash-attn with a KV cache, for the attention layers;
   - attn_gym's `recurrent_gdn`, already imported by `models/qwen3_5/gdn.py`, for
     the GDN layers;
   - a one-step conv update.

   Sampling uses `temperature` and `top_p`. Sequences that hit EOS go idle.
4. `model.reshard()`. The caches are freed.

Cache size per token of context, Qwen3.5 (full attention in 1 of 4 layers):

| model | K/V per token | GDN state per sequence |
|---|---|---|
| 2B | 12 KB | 19 MB (fp32) |
| 9B | 32 KB | 50 MB (fp32) |

A 128K-token prompt therefore holds 4 GB of K/V on the 9B, and long-context OPD fits
on one GH200.

Model changes are additive, with the training forward untouched:

- `forward_decode(x, cache)` on the attention and GDN modules;
- a cache object;
- a `prefill` flag that returns the caches.

TitanRL's `models/attention.py` and `models/gdn.py` are the reference for the same
modules.

**Lockstep.** All DP ranks run the same number of decode steps:

- FSDP is unsharded, so dense decode has no collectives.
- Under EP, every decoded token is an all-to-all.

Ranks all-reduce an "all done" flag every 64 steps and stop together, at
`max_new_tokens` at the latest. Idle sequences cost nothing on the attention and GDN
kernels, but still run the MLP; acceptable for M1.

**Train/sample mismatch.** The decode kernels differ from the training kernels
(recurrent vs chunked GDN, decode vs varlen attention). The sampled tokens' log-probs
therefore differ slightly from the training forward's. The loss uses the training
forward only, so a mismatch makes the samples marginally off-policy, not wrong.

- Logged per step: the mean |Δ log p| of the sampled tokens.
- The vendored batch-invariant flag (`train/parallel/batch_invariant.py`,
  `recurrent_gdn` training path) is for bitwise studies only, not the default.

### Packing and labels

Prompt + completion pairs are packed into `seq_len` rows with the SFT packer.

- Labels are the completion tokens through EOS, plus the prompt tokens when
  `prompt_kl_weight > 0`. A per-token weight tensor carries the two weights.
- Truncated completions (no EOS within `max_new_tokens`) are kept and counted.
- Ranks can end up with different numbers of rows, and FSDP needs the same number of
  forwards on every rank. The rank count is all-reduced to its maximum, and short
  ranks run a padding row with every label at -100.
- `global_tokens` normalisation handles the padding, as for SFT.

### Teacher

Built by `OPDTrainer._setup_teacher()` along the student's path:

1. `build_meta` with `tp=1`, the teacher's attention backend, `num_mtp_layers=0` and
   `moe_load_balance_coeff=0`.
2. `requires_grad_(False)`.
3. `cast_master_weights(..., "bfloat16")`.
4. `parallelize_qwen3_5` on the same `parallel_dims`, with no AC and
   `reshard_after_forward=True`.
5. `materialize`, then `load_hf(teacher_dir)`, then `eval()`.

Lifecycle:

- Not in the trainer's state dict, so checkpoints, resume and `dcp_to_hf` are
  unchanged.
- Self-distillation loads a second copy, because the student moves away from it.

Sharded teacher weights per GPU on 4-way FSDP:

- 2B: 1.2 GB;
- 9B: 4.5 GB.

### Loss (`train/step.py`, `train/loss.py`)

`forward_backward(..., teacher=None, opd=None)`, per micro-batch:

1. `teacher.preprocess_inputs(batch)`: the teacher's own masks, since its backend
   may differ.
2. Under no_grad, `teacher._skip_lm_head = True`, then the forward gives
   `teacher_hidden [T, D]`.

`teacher_hidden` is 64K × 2048 bf16 = 256 MB. The teacher's logits are never
materialised for a whole row: that would be 32 GB at 248K vocab.

`chunked_loss` splits `teacher_hidden` like the student's hidden states. Per chunk:

1. `logits = lm_head(h)`;
2. `teacher_logits = teacher_lm_head(h_t)`, under no_grad;
3. `loss = w_tok · (kl_weight · KL + ce_weight · CE)`, with the KL from
   `log_softmax` of both in fp32, summed over the vocabulary.

- The teacher's `lm_head` is unsharded once before the chunk loop and resharded
  after.
- MTP depths keep plain CE.
- Same normaliser as CE, so DP and accumulation scaling are unchanged.
- `kl_sum` gets a compiled twin, like `cross_entropy_sum`.
- Chunk memory: two `[T / loss_chunks, 248K]` fp32 blocks. Raise `loss_chunks` with
  `seq_len`.

`train/precompile.py` runs the teacher too.

### Logging

Added to the step line and to wandb:

- `loss_kl`, `loss_ce`;
- rollout time;
- generated tokens/s/GPU;
- mean and max completion length, EOS rate, prompts skipped;
- mean |Δ log p| (sample vs train);
- teacher forward time.

MFU counts the student only, and the line says so. Generation, teacher and student
time are logged separately, so the bottleneck is visible.

## Cost per step: estimate, to measure in M1

9B on one GH200 node: 64 sequences per GPU, 2K-token prompts, 2K new tokens.

- **Decode:** the weight-read floor is 18 GB / 4 TB/s = 4.5 ms per step. Without
  CUDA graphs, kernel launches probably push it to 10–20 ms, so 2K steps take
  20–40 s for 128K new tokens per GPU.
- **Student training on 256K tokens:** about 53 s at the measured 4.9K tok/s/GPU
  (16K, FSDP, MBS 4).
- **Teacher forward:** about 1/3 of that, 18 s.

Rollout is roughly 25–40% of the step. The 2B is cheaper in proportion: a 1.2 ms
floor and 15.5K tok/s/GPU training. M4 is triggered when rollout exceeds half the
step.

## Milestones

**M0. `train_opd.py` with `rollout = "dataset"` (KD).** No generation yet.

- Includes:
  - the entry point, `OPDConfig` and the `TRAIN_MODULE` launcher hook;
  - the teacher;
  - the KL loss;
  - the logging.
- Checks:
  1. Teacher = student, same backend: `loss_kl` is about 0 at step 0, and `loss_ce`
     equals a plain SFT run.
  2. Chunked loss with a teacher equals an unchunked reference, in value and in the
     gradient of the hidden states.
  3. `kl_sum` fkl and rkl match `torch.distributions`, and masked positions give 0.
  4. `train_qwen` on a 20-step SFT run matches the losses before the change.
  5. `models/tests/test_configs_parse.py` parses every toml.

M0 results, Qwen3.5-2B on one RTX PRO 6000, 8K rows, `onevision_instruct_samples`,
`configs/local/qwen3_5_2b_opd.toml`:

- Checks 2, 3, 5: `models/tests/test_distill_loss.py` and `test_configs_parse.py` pass.
- Check 4: `train_qwen` steps 1–2 are bit-identical to the pre-change code; later
  steps differ at the 3rd decimal place, as two runs of the old code do
  (non-deterministic kernels).
- Check 1: self-distillation step 1 gives `loss_kl` 0.0006 and `loss_ce` 2.0922,
  against SFT's 2.0921. Over 20 steps the KL then wanders up to 0.01–0.1: AdamW
  turns the near-zero KL's noise gradient into full lr-sized steps. The gradient
  of a KL scales like √KL (Pinsker), so `gnorm` stays CE-sized.
- KD, student Qwen3.5-2B-Base, teacher Qwen3.5-2B, 30 steps, lr 1e-5, warmup 5.
  The reference is the same batches at lr 0 (KL with no learning):

  | steps | fkl KL / lr-0 KL, mean per batch |
  |---|---|
  | 1–3 | 0.86 |
  | 4–10 | 0.48 |
  | 11–20 | 0.23 |
  | 21–30 | 0.26 |

  The `rkl` run falls likewise, from 0.19–0.28 to 0.05–0.09; its KL is a different
  measure, so it has no lr-0 ratio.
- Cost: 5.2–7.5K tok/s against 7–9.6K for SFT, so the teacher costs about 25%.
  Peak reserved memory is 83 GB against 76 GB.

### SFT vs KD (2026-11-09, `scripts/local_opd_compare.sh`)

Student Qwen3.5-2B-Base, 40M tokens per arm, everything identical except the loss:
- 1220 steps × 4 rows × 8K;
- WSD: 50 steps of warmup, then 20% cosine decay;
- lr 1e-5;
- same data order (`configs/local/opd_cmp_{sft,kd}.toml`).

The KD arm uses forward KL to Qwen3.5-2B, with `ce_weight = 0`. The held-out set is
100 rows of 8K from three tars kept out of training (`coco_train-00029`,
`ocr_train-00028/29`), evaluated at lr 0.

| model | held-out CE on the answers | held-out KL(teacher ‖ model) |
|---|---|---|
| base (Qwen3.5-2B-Base) | 1.106 | 0.282 |
| teacher (Qwen3.5-2B) | 1.348 | 0.0004 |
| SFT | **0.600** | 0.971 |
| KD | 1.300 | **0.026** |

Each arm optimises what it targets:

- **SFT** fits the dataset's answers: CE -46%. But it moves 3.4× further from the
  post-trained model than the base was.
- **KD** lands on the teacher: KL -91%. Its CE ends between the base and the
  teacher.
- The two metrics measure distance to the data and distance to the teacher. Neither
  says which model is better; that needs generation benchmarks on `hf_sft` and
  `hf_kd`.

Training:

- **SFT:** train CE 1.09 to 0.86. 9.1K tok/s, 76 min, 76.2 GB allocated / 90.4 GB
  reserved at peak.
- **KD:** train KL 0.090 to 0.039. 6.9K tok/s, 99 min (+30% for the teacher),
  82.2 GB allocated / 93.2 GB reserved at peak.
- The local driver does not set `expandable_segments`; KD fits on a 97.9 GB card
  without it.

#### Generation benchmarks (`scripts/local_bench.sh`, first 150 samples per task)

The run uses lmms-eval 0.7.1 on vLLM 0.19.1 with greedy decoding.

- The students answer after `<|im_start|>assistant\n`, as in training.
- The teacher uses its official template (non-thinking: an empty think block).

| model | AI2D | ChartQA | DocVQA | TextVQA | InfoVQA | RealWorldQA | MME perc. | MME cogn. |
|---|---|---|---|---|---|---|---|---|
| base | 68.0 | **63.3** | 88.9 | 66.7 | 69.4 | 63.3 | 123.6 | 67.5 |
| SFT | **80.7** | 56.0 | **94.0** | **81.3** | 72.5 | 68.0 | **140.9** | 92.5 |
| KD | 55.3 | 52.0 | 85.0 | 51.6 | 66.2 | 49.3 | 67.3 | 15.0 |
| teacher | 72.7 | 54.7 | 93.5 | 80.5 | **78.6** | **70.7** | 118.2 | **107.5** |

How to read the table:

- The first 150 samples are not a random subset: MME's come from a few categories.
  ±4 points is noise.
- OCRBench (14.1–14.4 for every model) is left out: its score is normalised to the
  full 1000 samples, so with 150 it saturates.

**KD's low scores are a prompt-format artifact, not a capability loss.** Share of
answers that open a `<think>` block:

| model | AI2D | ChartQA | DocVQA | TextVQA | InfoVQA | RealWorldQA | MME |
|---|---|---|---|---|---|---|---|
| base | 19% | 8% | 3% | 4% | 8% | 14% | 5% |
| SFT | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| KD | 34% | 23% | 11% | 35% | 21% | 39% | 49% |
| teacher | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

The cause:

1. The training template renders an answer without reasoning as
   `<|im_start|>assistant\n<answer>`, with no think block.
2. Qwen3.5-2B's native non-thinking format is
   `assistant\n<think>\n\n</think>\n\n<answer>`. Without the empty block, the
   teacher puts real mass on `<think>` at the first answer token.
3. Forward KL copies that whole distribution into the student. SFT's one-hot target
   never does.
4. At eval, greedy decoding then opens a think block on up to half of the prompts,
   and the long reasoning fails answer extraction.

This was a pitfall of the KD setup and is not specific to this comparison: both
models must see a prompt format the teacher is native to.

Fix: render every reasoning-free answer with the empty think block (student and
teacher see the same tokens), and evaluate with `enable_thinking=false`.

**M1. Native decode, text, dense; `rollout = "student"`.**

- Includes:
  - prefill caches and the decode loop for `qwen3_5` and `qwen3_5_text`;
  - lockstep;
  - packing with padding rows.
- Checks:
  1. Greedy tokens from the native decode equal vLLM 0.19.1 (local `vllm` env) on 32
     prompts with Qwen3.5-2B, up to the first bf16 tie.
  2. Mean |Δ log p| between sample and train is below 1e-2 nats.
  3. Fixed seed: two runs give the same rollouts.
  4. Measure the cost split above.

**M2. Image prompts, MoE + EP, DSA student.**

- Vision runs in prefill only.
- EP decode in lockstep.
- The DSA student additionally needs an indexer-key cache and top-k selection per
  decoded token. It lands with the DSA merge, not on main first.

**M3. Long-context OPD.**

- Prompts of 128K and more, with `prompt_kl_weight > 0`. A 1K answer to a 512K
  prompt is 0.2% of the tokens.
- Mostly config and memory, and it rides on the long-context SFT work. Rows must fit
  one GPU until there is CP.

**M4. Faster rollouts, only if rollout exceeds half the step.**

- Capture the decode step in CUDA graphs: the cache is static per rollout.
- Otherwise TitanRL's vLLM wrapper of the same model definitions. It needs the
  nightly stack on JUPITER/MN5 (aarch64 wheels not checked).

## Limits

- **TP:** the KL needs a vocab-parallel log-softmax for both models. Add it when the
  35B needs TP.
- **CP:** none for Qwen3.5, so a row, prompt included, fits one GPU.
- **Different tokenizers:** not supported.
- **Rollout memory and time:** rollouts are synchronous and colocated, so generation
  and training do not overlap. Asynchronous generation (TitanRL's
  `max_offpolicy_steps`) is out of scope.
- **Multi-turn and tool rollouts:** single-turn completions only.

## Effort

| milestone | effort | depends on |
|---|---|---|
| M0 | 3–4 days | nothing |
| M1 | 1.5–2 weeks | M0 |
| M2 | ~1 week | M1; DSA merge for the DSA student |
| M3 | days on top of long-context SFT | M1, long-context SFT |
| M4 | 1 week (graphs) to 3 weeks (vLLM) | M1 measurements |

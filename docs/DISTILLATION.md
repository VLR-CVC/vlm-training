# Plan: knowledge distillation and on-policy distillation

Status: plan, not implemented. A general trainer feature, independent of any one
project. A frozen teacher model supervises the student with a per-token divergence
over the full vocabulary, in place of or alongside the next-token loss.

## Use cases

The teacher is any model that shares the student's tokenizer and vocabulary.

| case | teacher | student |
|---|---|---|
| self-distillation after an architecture change: sparse attention, context extension, attention-kind swaps on `qwen3_5_text` | the original checkpoint (`training.model_dir` itself), possibly with another `attn_backend` | the same weights with the changed architecture |
| capability-preserving fine-tune: anchor a domain SFT to the starting model | the starting checkpoint | the model being fine-tuned, `ce_weight > 0` |
| compression | Qwen3.5-9B / 35B-A3B | Qwen3.5-2B / 0.8B |
| on-policy distillation (OPD, phase 1) | any of the above | samples its own completions |

## Decision: this trainer, no RL framework

Distillation, on-policy included, is not RL. There is no reward, no advantage, no
value model and no clipping. You run the teacher and the student on the same tokens
and minimise a divergence at every position (GKD, Agarwal et al. 2024). Off-policy
needs nothing the trainer lacks; on-policy needs only generation.

Moving to verl, slime, NeMo-RL, prime-rl or TRL would mean porting our models to
their trainer and their rollout engine:

- Qwen3.5 VLM with GatedDeltaNet;
- Qwen3-VL;
- the MoE with EP;
- the kernels.

It would also mean giving up the long-context machinery:

- packing;
- selective AC;
- chunked loss;
- FSDP2/EP;
- SR AdamW.

Those frameworks pay off for many long generations per step, such as reasoning or
agent traces. Revisit if that becomes the main workload.

No copy of `train/train_qwen.py`, because a fork drifts.

- **Phase 0** is a config section plus changes to the loss and step functions.
- **Phase 1** is a `Trainer` subclass that overrides only `batch_generator`.

Model definitions and parallelism code do not change. The one exception is the
optional native-generation path in phase 1 (engine A), which adds a decode cache.

## Phase 0: off-policy distillation

### Config: `[distill]` in `train/config.py`

A new `Distill` dataclass, added to `Config` as `distill`. Distillation is off
unless `teacher_dir` is set.

```toml
[distill]
teacher_dir = ""            # HF snapshot; training.model_dir = self-distillation
teacher_model_config = ""   # empty: teacher_dir/config.json
teacher_attn_backend = ""   # empty: model.attn_backend
divergence = "fkl"          # fkl: KL(teacher || student); rkl: KL(student || teacher)
kl_weight = 1.0
ce_weight = 0.0             # next-token loss on the labels; logged even at 0
```

Startup checks:

- the divergence name is valid;
- `tp_size = 1` and `loss_chunks > 1` (see below);
- the teacher and student vocab sizes are equal. They live in `vocab_size` or
  `text_config.vocab_size`, depending on the config.

### Teacher: `Trainer._setup_teacher()` in `train/train_qwen.py`

The teacher follows the same build path as `_setup_model`, in this order:

1. `build_meta` with `tp=1`, the teacher's `attn_backend`, `num_mtp_layers=0` and
   `moe_load_balance_coeff=0`. The coefficient is 0 so the teacher gets no bias
   buffer and no optimizer hook.
2. `requires_grad_(False)`.
3. `cast_master_weights(teacher, "bfloat16")`: bf16 storage, since no optimizer
   ever reads it.
4. `parallelize_qwen3_5` on the same `parallel_dims`:
   - same `data_parallel` mode and `compile`;
   - no selective AC;
   - `reshard_after_forward=True`.
5. `materialize`, then `load_hf(teacher_dir)`, then `eval()`.

Lifecycle:

- Stored as `self.teacher`, which is `None` when distillation is off.
- Not part of the trainer's state dict, so checkpoints, resume and HF conversion
  are unchanged.
- Log the `[distill]` fields to wandb with the rest of the config.

Memory is the teacher's weights split over the FSDP degree:

| teacher | weights | per GPU on 4-way FSDP |
|---|---|---|
| 2B | 4.6 GB | about 1.2 GB |
| 9B | 18 GB | 4.5 GB |

The teacher's no-grad activations are freed layer by layer. Its forward adds about
a third of a training step's FLOPs. `flops_estimation` does not count it, so MFU
reads low; note that in the logs.

### Step: `train/step.py::forward_backward`

New keyword arguments `teacher=None` and `distill=None`. For each micro-batch,
before the student forward:

- Run `teacher.preprocess_inputs(batch, ...)`. This is the teacher's own
  preprocessing, because its attention backend may build different masks.
  `preprocess_inputs` copies the dict, so the student's call is unaffected.
- Set `teacher._skip_lm_head = True`.
- Under `torch.no_grad()`, the teacher forward produces `teacher_hidden [T, D]`.
- Pass `teacher_hidden`, `teacher.lm_head` and the weights to `chunked_loss`.

`teacher_hidden` is small: 64K x 2048 in bf16 is 256 MB. Never materialise the
teacher's logits for a whole row: 64K x 248K in bf16 is 32 GB.

`train/precompile.py` must pass the teacher too, so its graphs compile before step 1.

### Loss: `train/loss.py`

`kl_sum(logits, teacher_logits, labels, reverse)`:

- `log_softmax` of both in fp32, then `F.kl_div(..., log_target=True)` summed over
  the vocabulary.
- Summed over the positions where `labels != -100`.
- A compiled twin, as for `cross_entropy_sum`.

`chunked_loss(..., teacher_hidden, teacher_lm_head, kl_weight, ce_weight, reverse_kl)`:

- Split `teacher_hidden` into the same chunks as the student.
- Per chunk, on the main output only: `logits = lm_head(h)`, then
  `teacher_logits = teacher_lm_head(h_t)` under no_grad, then
  `chunk_loss = ce_weight * CE + kl_weight * KL`.
- MTP depths keep plain CE.
- Unshard the teacher's FSDP `lm_head` once before the chunk loop and reshard it
  after, as the student's head already is. Otherwise every chunk all-gathers it
  again: 0.5–1 GB per gather for 2B/9B.
- Normaliser: the same labelled positions and `global_tokens` as CE, so DP and
  gradient-accumulation scaling are unchanged.
- Metrics: `loss_ce` and `loss_kl` as this rank's per-token means over the
  micro-batch. Rename the `"  mtp: "` label in `train/metrics.py` to
  `"  loss: "`, since the dict is no longer MTP-only.

Which tokens are supervised: the KL uses the CE label mask. Data with assistant-only
labels distils only the answers; `data.pretrain = true` labels, and so distils, every
token. A separate KL mask (all non-padding tokens of chat data) is an extension if
needed.

### Limits and their upgrade paths

- **Memory per chunk:** two `[T / loss_chunks, vocab]` fp32 blocks plus softmax
  temporaries. With 4K tokens per chunk and Qwen3.5's 248K vocab that is about
  4 GB per block. Raise `loss_chunks` at long `seq_len`; the docstring says so.
- **TP:** under `tp > 1`, logits are vocab-sharded. A vocab-parallel KL needs the
  all-reduced max and logsumexp of `_LossParallelCrossEntropy`, for both models.
  Add it when the 35B needs TP.
- **Different vocabularies:** not supported; that needs cross-tokenizer distillation.
- **Context parallelism:** not supported for Qwen3.5 yet, for the teacher as for the
  student.

### Checks

1. Teacher and student are the same checkpoint with the same backend: `loss_kl` is
   0 at step 0, up to bf16 noise, and `loss_ce` matches a plain run.
2. `chunked_loss` with a teacher matches an unchunked reference on a short row, in
   value and in the gradient of the hidden states.
3. Unit test of `kl_sum`: fkl and rkl against `torch.distributions` on random
   logits, and masked positions contribute 0.
4. `models/tests/test_configs_parse.py` still parses every toml with the new section.

## Phase 1: on-policy distillation

Each step:

1. Prompts.
2. The student generates completions.
3. Teacher and student forward on prompt + completion (the phase 0 path).
4. Loss on the completion tokens.

Prompt tokens can also carry a teacher-forced KL, with their own weight. Long
prompts with short answers need this: a 1K answer to a 512K prompt is 0.2% on-policy
tokens.

### `train/train_opd.py`

`OPDTrainer(Trainer)` overrides `batch_generator`. Configured from new `[distill]`
fields:

- `mode = "onpolicy"`;
- `max_new_tokens`, `temperature`, `samples_per_prompt`;
- `prompt_kl_weight`;
- `rollout_refresh_steps`, for engine B.

Labels mark the completion, plus the prompt when `prompt_kl_weight > 0`. The step,
loss, optimizer, checkpointing and logging are inherited unchanged.

### Prompts

A loader that yields prompts:

- the chat template up to the assistant turn, without labels;
- one sample per row, no packing during generation;
- prompt + completion packed again for the training forward.

For multimodal prompts, reuse the energon cookers in a prompt-only mode; for text,
a jsonl reader.

### Generation engine, in order

- **B. HF `generate`** on an export from `utils/dcp_to_hf.py`, refreshed every
  `rollout_refresh_steps`, so rollouts are semi-on-policy. Slow, but it needs no
  model changes and validates the loop.
- **A. Native generation in our models**, if B is too slow. It adds decode caches:
  - KV for the full-attention layers;
  - recurrent and conv state for GatedDeltaNet;
  - any architecture-specific caches.

  `models/qwen3_5/gdn.py` already has a state-cache recurrent variant. Unshard
  the model once before generating: FSDP all-gathers per decoded token would
  dominate. The 9B is 18 GB of resident weights.
- **C. vLLM/SGLang server**: only together with the framework switch above.

### Checks

- Greedy generation with engine A gives the same tokens as HF `generate`.
- The student log-probs from the training forward match the ones recorded at
  sampling time.

## Effort

| item | effort | blocks |
|---|---|---|
| phase 0 | 2–3 days | nothing |
| phase 1, engine B | ~1 week | phase 0 |
| engine A | 2–3 weeks | only if B is too slow |
| TP KL, CP | when needed | 35B with TP; rows that do not fit one GPU |

# Qwen3.5-9B scaling tests on JUPITER

Weak scaling: per-GPU work is fixed, the global batch grows with the node count.
A flat `tok/s/GPU` column means perfect scaling; the number that drops is the
one worth chasing.

```bash
./scripts/scaling/submit_sweep.sh              # 16 32 64 128 256 nodes
python scripts/scaling/collect.py logs_scaling/<sweep id>
```

| file | what it does |
|---|---|
| `jup_scaling.sbatch` | one point on the curve. Node count comes from `sbatch --nodes`, never from a directive. |
| `submit_sweep.sh` | creates the log dirs, submits one job per point, records job ids |
| `collect.py` | parses the logs into a table + CSV |
| `../../configs/jupiter/qwen3_5_9b.toml` | the default run config (`SCALING_CONFIG` overrides it) |

## Running it

```bash
# the default sweep
./scripts/scaling/submit_sweep.sh

# a custom set of points
./scripts/scaling/submit_sweep.sh 4 8 16 32

# see the sbatch lines without submitting anything
SCALING_DRY_RUN=1 ./scripts/scaling/submit_sweep.sh
```

Knobs, all environment variables:

| variable | default | |
|---|---|---|
| `SCALING_CONFIG` | `configs/jupiter/qwen3_5_9b.toml` | swap in `configs/jupiter/qwen3_vl_2b.toml` |
| `SCALING_ACCOUNT` | `open-sci-mm` | also `jureap59`, `reformo` |
| `SCALING_SWEEP_ID` | timestamp | names the log directory |
| `SCALING_LOG_ROOT` | `logs_scaling/` | |
| `SCALING_OUTPUT_ROOT` | `/e/scratch/open-sci-mm/$USER/scaling` | where the throwaway checkpoint goes |
| `SCALING_KEEP_CHECKPOINT` | `0` | `1` keeps the final checkpoint |

Jobs are independent. The 16-node point will run long before the 256-node one
clears the queue, and `collect.py` reads whatever has finished.

## Reading the results
- **Median, not mean.** One straggler step at 256 nodes moves a mean far more
  than it reflects what the run sustains.
- **`--skip 10` by default**, because the first steps are `torch.compile` and the
  inductor autotuner, not training. Raise it if the step time has not settled;
  the per-step lines in `log.out` show where it flattens.
- **`data%`** is the share of step time spent waiting on the dataloader. If
  efficiency falls and `data%` climbs with it, the problem is data, not NCCL.
- **`speedup`/`eff%`** are relative to the smallest point in the sweep, so a
  sweep starting at 16 reads 16 -> `1.00x`. Comparing two sweeps means comparing
  the `tok/s/GPU` column, not the efficiency column.
- A point with no usable steps is named explicitly at the bottom rather than
  dropped. Look in `<N>n/errors/rank_*.err`.

## What the config does differently
- `seq_len = 10240`
- `total_steps = 40`, enough for ~30 measured steps after warmup
- `save_steps = 100000`, so nothing checkpoints mid-run
- `repeat = true`, so a 40-step run cannot exhaust a shard at high node counts
  and cut the measurement short
- `save_dataloader_state = false`, which otherwise shows up in the step time

`tp_size = 4` keeps tensor parallelism inside a node, so adding nodes adds data
parallelism only. That is the axis being measured.

## Things that will bite you

**The final checkpoint.** `end_run` writes one unconditionally — a 9B model plus
AdamW state is ~100 GB, once per sweep point, never read. `jup_scaling.sbatch`
deletes it after the run. If a job hits its wall clock the deletion never runs,
so check `$SCALING_OUTPUT_ROOT` after a sweep that timed out.

**Log directories.** SLURM does not create the directories in an `--output`
path; a missing one kills the job at launch with nothing written to explain it.
`submit_sweep.sh` creates them, which is the main reason to use it rather than
calling `sbatch` by hand.

**`module load CUDA/13`.** FLA hands the gated delta rule to flash_qla whenever
it imports, and flash_qla JIT-compiles through tilelang, which needs `nvcc` on
`PATH`. Without it the job dies with `ValueError: No CUDA or HIP or MPS
available on this system` even though the GPUs are fine. See INSTALL.md.

**Queue time.** 256 nodes on `booster` is not instant. Submit the whole sweep at
once and collect later; do not sit on an interactive allocation.

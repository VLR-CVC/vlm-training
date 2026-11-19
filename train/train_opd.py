"""On-policy distillation (docs/OPD.md): `python -m train.train_opd --config ...`.

M0: the completions are the dataset's answers (off-policy distillation). The
trainer is `Trainer` plus a frozen teacher; `train.step.forward_backward` runs it
and `train.loss.chunked_loss` turns the loss into a KL against it.
"""
import faulthandler
import json
import signal
import sys
from pathlib import Path

import torch
import wandb

from train.config import OPDConfig
from train.config_manager import ConfigManager
from train.logger import init_logger, logger
from train.train_qwen import Trainer
from train.utils import cast_master_weights


def _vocab_size(hf_config: dict) -> int:
    return hf_config.get("vocab_size") or hf_config["text_config"]["vocab_size"]


class OPDTrainer(Trainer):
    def __init__(self, cfg: OPDConfig):
        opd = cfg.opd
        if opd.teacher_dir in ("", "NULL"):
            raise ValueError("set opd.teacher_dir (training.model_dir for self-distillation)")
        if opd.divergence not in ("fkl", "rkl"):
            raise ValueError(f"opd.divergence must be fkl or rkl, got {opd.divergence!r}")
        if cfg.training.tp_size != 1:
            raise NotImplementedError("distillation under TP needs a vocab-parallel KL")
        if cfg.training.loss_chunks <= 1:
            raise ValueError("distillation needs training.loss_chunks > 1")
        super().__init__(cfg)
        self.opd_args = opd
        self._setup_teacher()
        if self.if_log_rank():
            logger.info(opd)
            wandb.config.update({f"opd/{k}": v for k, v in vars(opd).items()})

    def _setup_teacher(self):
        """Student's build path, frozen: meta -> bf16 -> FSDP (resharded after
        forward, no AC) -> materialise -> load HF. Not in the trainer's state dict."""
        from models.qwen3_5.checkpoint import build_meta, load_hf, materialize
        from train.parallel.parallelize import parallelize_qwen3_5

        opd = self.opd_args
        config_path = (
            Path(opd.teacher_dir) / "config.json"
            if opd.teacher_model_config in ("", "NULL")
            else Path(opd.teacher_model_config)
        )
        hf_config = json.loads(config_path.read_text())
        if _vocab_size(hf_config) != _vocab_size(self.hf_config):
            raise ValueError(
                f"teacher vocab {_vocab_size(hf_config)} != student vocab "
                f"{_vocab_size(self.hf_config)}"
            )
        teacher = build_meta(
            config_path,
            seq_len=int(self.data_args.seq_len),
            tp=1,
            attn_backend=opd.teacher_attn_backend or self.model_args.attn_backend,
            decoder_mask=self.model_args.decoder_mask,
            num_mtp_layers=0,
            # no bias buffer, no optimizer hook
            moe_load_balance_coeff=0.0,
            ep=self.training_args.ep_size,
        )
        teacher.requires_grad_(False)
        cast_master_weights(teacher, "bfloat16")
        parallelize_qwen3_5(
            teacher,
            self.parallel_dims,
            mode=self.training_args.data_parallel,
            compile=self.training_args.compile,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            reshard_after_forward=True,
        )
        materialize(teacher, self.device)
        load_hf(teacher, opd.teacher_dir)
        teacher.eval()
        self.teacher = teacher
        logger.info(f"teacher from {opd.teacher_dir} ({config_path})")


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    config = ConfigManager(OPDConfig).parse_args(sys.argv[1:])
    init_logger()
    torch.manual_seed(42)
    OPDTrainer(config).train()

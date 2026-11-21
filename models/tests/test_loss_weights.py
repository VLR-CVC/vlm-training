"""`data.loss_unit` / `loss_sqrt`: the weighted objective equals ElliotVL's Megatron
reduction (mean over rows/samples of their own weighted means), and the logged value
equals its `lm loss` (sum(w CE) / sum(w))."""

from types import SimpleNamespace

import torch

from data.energon_dataloader import PackedBatchEncoder


def _sample(n_sup, length):
    labels = torch.full((length,), -100)
    labels[:n_sup] = 1
    return SimpleNamespace(labels=labels, length=length)


def test_loss_weights():
    rows = [[_sample(3, 5), _sample(10, 12)], [_sample(1, 4)], []]  # last row: padding only
    row_length = 20
    ce = torch.rand(row_length * len(rows))  # one CE per token slot
    for unit in ("token", "row", "sample"):
        for sqrt in (False, True):
            enc = SimpleNamespace(loss_unit=unit, loss_sqrt=sqrt, row_length=row_length)
            out = PackedBatchEncoder._loss_weights(enc, rows)
            objective = (ce * out["loss_weights"]).sum() / out["loss_norm"]
            logged = (ce * out["log_weights"]).sum() / out["log_norm"]

            # reference: walk the slots sample by sample
            pos, per_row, per_sample, wsum, wce = 0, [], [], 0.0, 0.0
            for row in rows:
                rw, rce = 0.0, 0.0
                for s in row:
                    n = int((s.labels != -100).sum())
                    b = n**-0.5 if sqrt else 1.0
                    c = ce[pos : pos + n]
                    per_sample.append(c.mean())
                    rw, rce = rw + n * b, rce + b * c.sum()
                    pos += s.length
                pos += row_length - sum(s.length for s in row)
                if rw:
                    per_row.append(rce / rw)
                wsum, wce = wsum + rw, wce + rce
            want = {"token": wce / wsum, "row": torch.stack(per_row).mean(),
                    "sample": torch.stack(per_sample).mean()}[unit]
            torch.testing.assert_close(objective, want)
            torch.testing.assert_close(logged, wce / wsum)


if __name__ == "__main__":
    test_loss_weights()
    print("ok")

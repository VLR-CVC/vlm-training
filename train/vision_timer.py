from __future__ import annotations

import torch

class VisionTimer:
    def __init__(self) -> None:
        self.enabled = False
        self._cur: list[dict] = []
        self._prev: list[dict] = []
        self._cur_counts = [0, 0]  # patches, sum over images of patches^2 (attention cost)
        self._prev_counts = [0, 0]

    def new_step(self) -> None:
        if self.enabled:
            self._prev, self._cur = self._cur, []
            self._prev_counts, self._cur_counts = self._cur_counts, [0, 0]
        else:  # nothing stale survives a switch-off
            self._prev, self._cur = [], []
            self._prev_counts, self._cur_counts = [0, 0], [0, 0]

    @staticmethod
    def _event() -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def forward_start(self, grids) -> dict | None:
        """``grids``: list of [t, h, w] in patches. Returns the record to pass on."""
        if not self.enabled or torch.compiler.is_compiling():
            return None
        rec = {"f0": self._event()}
        self._cur.append(rec)
        for t, h, w in grids:
            self._cur_counts[0] += t * h * w
            self._cur_counts[1] += t * (h * w) ** 2
        return rec

    def forward_end(self, rec: dict | None) -> None:
        if rec is not None:
            rec["f1"] = self._event()

    def backward_marks(self, rec: dict | None, out_side: torch.Tensor, in_side: torch.Tensor) -> None:
        """Record when backward reaches ``out_side`` (start) and ``in_side`` (end)."""
        if rec is None or not (out_side.requires_grad and in_side.requires_grad):
            return

        def mark(name):
            def hook(grad):
                rec[name] = self._event()  # autograd runs the hook on the op's stream
            return hook

        out_side.register_hook(mark("b0"))
        in_side.register_hook(mark("b1"))

    def collect(self) -> tuple[float, float, int, int]:
        """Previous step's (forward ms, backward ms, patches, sum patches^2) on this rank."""
        fwd = bwd = 0.0
        for rec in self._prev:
            if "f1" in rec:
                rec["f1"].synchronize()
                fwd += rec["f0"].elapsed_time(rec["f1"])
            if "b0" in rec and "b1" in rec:
                rec["b1"].synchronize()
                bwd += rec["b0"].elapsed_time(rec["b1"])
        return fwd, bwd, self._prev_counts[0], self._prev_counts[1]


VISION_TIMER = VisionTimer()

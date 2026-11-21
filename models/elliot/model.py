from dataclasses import dataclass

import torch

from models.qwen3_5.model import Qwen35Model

IMAGE_POSITIONS = ("mrope", "1d")
DECODER_MASKS = ("causal_doc", "causal")

class ElliotModel(Qwen35Model):
    """Differences are defined in `elliot/configs.py`"""
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35Model.Config):
        image_positions: str = "mrope"
        decoder_mask: str = "causal_doc"
        row_len: int = 0  # packed row length, used by decoder_mask "causal"

    def __init__(self, config: Config):
        if config.image_positions not in IMAGE_POSITIONS:
            raise ValueError(f"image_positions {config.image_positions!r} not in {IMAGE_POSITIONS}")
        if config.decoder_mask not in DECODER_MASKS:
            raise ValueError(f"decoder_mask {config.decoder_mask!r} not in {DECODER_MASKS}")
        if config.decoder_mask == "causal" and (config.image_positions != "1d" or config.row_len <= 0):
            raise ValueError('decoder_mask "causal" needs image_positions "1d" and row_len > 0')
        super().__init__(config)

    def preprocess_inputs(self, input_dict, *, parallel_dims=None):
        if self.config.image_positions == "1d":
            input_dict = {k: v for k, v in input_dict.items() if k != "mrope_positions"}
        if self.config.decoder_mask == "causal":
            # positions restart only at row starts: one document per row for the masks
            pos = input_dict["positions"]
            if pos.shape[0] % self.config.row_len:
                raise ValueError(f"{pos.shape[0]} tokens are not whole rows of {self.config.row_len}")
            input_dict = {**input_dict, "positions": torch.arange(pos.shape[0], device=pos.device) % self.config.row_len}
        return super().preprocess_inputs(input_dict, parallel_dims=parallel_dims)

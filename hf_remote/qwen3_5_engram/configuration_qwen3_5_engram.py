"""Config of the Qwen3.5 + Engram text model (trust_remote_code).

The Qwen3.5 text config plus one ``engram`` dict, written by
`utils/dcp_to_hf.py`:

    {"layers": [2], "orders": [2, 3], "heads_per_order": 8,
     "rows_per_head": 262144, "dim": 1024, "seed": 0, "gate_sqrt": true,
     "pad_id": 248044, "blocked_token_ids": [248056, 248057]}

Every field there feeds `engram_hash.build_ngram_hash_spec` exactly as in
training, so the exported model addresses the same rows.
"""

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


class Qwen3_5EngramConfig(Qwen3_5TextConfig):
    model_type = "qwen3_5_engram"

# Route the vendored DSA sparse-attention forward through the grouped tensor-core
# kernel (see sparse_attention.py; DSA_GROUPED_SPARSE_ATTN=0 turns it off).
try:
    from models.common.dsa import sparse_attention as _sparse_attention
except ImportError:  # no Triton: the vendored module falls back to PyTorch anyway
    pass
else:
    _sparse_attention.install()

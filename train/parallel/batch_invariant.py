# Vendored from torchtitan b21f7d43e: the batch-invariant flag of
# torchtitan/distributed/utils.py. Nothing here enables it yet; the flag exists so
# vendored model code (GDN recurrent path) keeps its upstream branches.

_batch_invariant_enabled = False


def is_in_batch_invariant_mode() -> bool:
    """Return whether batch-invariant mode is active."""
    return _batch_invariant_enabled

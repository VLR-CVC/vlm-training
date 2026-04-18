"""Shared pytest configuration for models/tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "cuda_only: test requires a CUDA device",
    )

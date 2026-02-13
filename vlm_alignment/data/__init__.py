"""Data loading and generation.

Imports are lazy to avoid heavy dependencies at package import time.
"""


def __getattr__(name):
    if name == "DataGenerator":
        from .synthetic import DataGenerator
        return DataGenerator
    elif name == "VLMDataset":
        from .dataset import VLMDataset
        return VLMDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

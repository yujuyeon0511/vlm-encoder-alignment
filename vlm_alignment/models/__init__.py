"""Model loading and embedding extraction.

Imports are lazy to avoid requiring torch at package import time.
"""


def __getattr__(name):
    if name == "VisionEncoderManager":
        from .vision_encoders import VisionEncoderManager
        return VisionEncoderManager
    elif name == "LLMManager":
        from .llm_loaders import LLMManager
        return LLMManager
    elif name == "create_projector":
        from .projectors import create_projector
        return create_projector
    elif name == "train_projector":
        from .projectors import train_projector
        return train_projector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

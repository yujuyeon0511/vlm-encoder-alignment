"""Central configuration loader with environment variable overrides."""

import os
import yaml
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_CACHE = None


def get_project_root() -> Path:
    return _PROJECT_ROOT


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config.yaml with environment variable overrides.

    Priority: environment variables > config.yaml > defaults.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Environment variable overrides
    if os.environ.get("VLM_DATA_ROOT"):
        cfg["data"]["root"] = os.environ["VLM_DATA_ROOT"]

    if os.environ.get("VLM_OUTPUT_DIR"):
        cfg["defaults"]["output_dir"] = os.environ["VLM_OUTPUT_DIR"]

    if config_path == _PROJECT_ROOT / "config.yaml":
        _CONFIG_CACHE = cfg

    return cfg


def get_data_root() -> Path:
    """Get data root directory. Falls back to sample_data/ if not configured."""
    cfg = load_config()
    root = cfg["data"].get("root")
    if root and Path(root).exists():
        return Path(root)
    return _PROJECT_ROOT / cfg["data"]["sample_dir"]


def get_model_id(category: str, name: str) -> str:
    """Get HuggingFace model ID from config.

    Args:
        category: 'vision_encoders' or 'llms'
        name: model short name (e.g., 'clip', 'llama3')
    """
    cfg = load_config()
    models = cfg["models"].get(category, {})
    if name.lower() not in models:
        raise ValueError(
            f"Unknown model '{name}' in '{category}'. "
            f"Available: {list(models.keys())}"
        )
    return models[name.lower()]


def get_output_dir() -> Path:
    cfg = load_config()
    out = _PROJECT_ROOT / cfg["defaults"]["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_device() -> str:
    """Get device from config. 'auto' resolves to cuda/cpu."""
    import torch
    cfg = load_config()
    device = cfg["defaults"].get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

"""Unified plot style configuration (SciLinkBERT-inspired)."""

import matplotlib.pyplot as plt

# Model colors
COLORS = {
    "CLIP": "#3b7faf",
    "SigLIP": "#ff913d",
    "DINOv2": "#4ba0b1",
    "InternViT": "#8B5CF6",
    "PaliGemma": "#EC4899",
    "LLaVA": "#9cb35e",
    "Qwen": "#df4054",
    "LLaMA": "#3B82F6",
    "Gemma": "#10B981",
    "Gemma3": "#059669",
    "InternLM": "#F59E0B",
    "default": "#eec662",
}

DATA_TYPE_COLORS = {
    "chart": "#3b7faf",
    "table": "#ff913d",
    "text": "#4ba0b1",
    "visualization": "#9cb35e",
    "math": "#df4054",
}

GRID_COLOR = "#757171"
BORDER_COLOR = "#333333"

FONT_CONFIG = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
}

FIGURE_CONFIG = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": BORDER_COLOR,
    "axes.linewidth": 1.5,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.5,
    "grid.linewidth": 1.0,
    "grid.linestyle": "--",
}


def apply_style():
    """Apply unified plot style."""
    config = {}
    config.update(FONT_CONFIG)
    config.update(FIGURE_CONFIG)
    plt.rcParams.update(config)


def get_model_color(model_name: str) -> str:
    """Get color for a model by name."""
    name = model_name.lower()
    if "clip" in name and "sig" not in name:
        return COLORS["CLIP"]
    elif "siglip" in name or "sig" in name:
        return COLORS["SigLIP"]
    elif "dino" in name:
        return COLORS["DINOv2"]
    elif "internvit" in name:
        return COLORS["InternViT"]
    elif "paligemma" in name:
        return COLORS["PaliGemma"]
    elif "llava" in name:
        return COLORS["LLaVA"]
    elif "qwen" in name:
        return COLORS["Qwen"]
    elif "llama" in name:
        return COLORS["LLaMA"]
    elif "gemma3" in name or "gemma-3" in name:
        return COLORS["Gemma3"]
    elif "gemma" in name:
        return COLORS["Gemma"]
    elif "internlm" in name:
        return COLORS["InternLM"]
    return COLORS["default"]


def get_data_type_color(data_type: str) -> str:
    return DATA_TYPE_COLORS.get(data_type.lower(), COLORS["default"])


def style_axis(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """Apply consistent styling to an axis."""
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if grid:
        ax.grid(True, alpha=0.5, linewidth=1.0, linestyle="--", color=GRID_COLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color(BORDER_COLOR)


def create_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """Create a figure with consistent styling."""
    apply_style()
    if figsize is None:
        if nrows == 1 and ncols == 1:
            figsize = (8, 6)
        elif nrows == 1:
            figsize = (6 * ncols, 5)
        elif ncols == 1:
            figsize = (8, 5 * nrows)
        else:
            figsize = (6 * ncols, 5 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, **kwargs)


def save_figure(fig, path, dpi=300, formats=None):
    """Save figure in multiple formats."""
    import os

    if formats is None:
        formats = ["png"]
    base = os.path.splitext(path)[0]
    for fmt in formats:
        save_path = f"{base}.{fmt}"
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

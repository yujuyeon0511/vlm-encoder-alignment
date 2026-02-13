"""Alignment visualization: CKA heatmaps, projector comparisons, and error analysis.

Consolidates visualizations from encoder_llm_alignment.py, fair_comparison_v2.py,
error_attribution.py, structural_info.py, and multi_encoder_comparison.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, get_data_type_color,
    style_axis, create_figure, save_figure, COLORS,
)


def plot_cka_comparison(
    cka_scores: Dict[str, float],
    title: str = "CKA Score by Encoder",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing CKA scores across encoders.

    Args:
        cka_scores: {encoder_name: cka_score}
        title: Chart title
        output_path: Save path
    """
    fig, ax = create_figure()
    names = list(cka_scores.keys())
    values = list(cka_scores.values())
    colors = [get_model_color(n) for n in names]

    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    style_axis(ax, title=title, ylabel="CKA Score")
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_cka_by_data_type(
    scores_by_type: Dict[str, Dict[str, float]],
    title: str = "CKA Score by Data Type",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Grouped bar chart of CKA scores by data type.

    Args:
        scores_by_type: {data_type: {encoder_name: cka_score}}
    """
    fig, ax = create_figure(figsize=(10, 6))

    data_types = list(scores_by_type.keys())
    encoders = list(next(iter(scores_by_type.values())).keys())
    n_types = len(data_types)
    n_enc = len(encoders)
    width = 0.8 / n_enc
    x = np.arange(n_types)

    for i, enc in enumerate(encoders):
        values = [scores_by_type[dt].get(enc, 0) for dt in data_types]
        offset = (i - n_enc / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=enc.upper(),
                      color=get_model_color(enc), edgecolor="black", linewidth=1)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in data_types])
    ax.legend()
    style_axis(ax, title=title, ylabel="CKA Score")

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_projector_comparison(
    results: Dict[str, Dict[str, float]],
    title: str = "Projection Strategy Comparison",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare projector types across encoders.

    Args:
        results: {encoder_name: {projector_type: loss}}
    """
    fig, ax = create_figure(figsize=(10, 6))

    encoders = list(results.keys())
    proj_types = list(next(iter(results.values())).keys())
    n_enc = len(encoders)
    n_proj = len(proj_types)
    width = 0.8 / n_proj
    x = np.arange(n_enc)

    hatches = ["", "//", "\\\\", "xx"]
    for i, pt in enumerate(proj_types):
        values = [results[enc].get(pt, 0) for enc in encoders]
        offset = (i - n_proj / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=pt,
               color=[get_model_color(e) for e in encoders],
               edgecolor="black", linewidth=1,
               hatch=hatches[i % len(hatches)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([e.upper() for e in encoders])
    ax.legend(title="Projector")
    style_axis(ax, title=title, ylabel="MSE Loss")

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_cka_heatmap(
    encoder_names: List[str],
    llm_names: List[str],
    scores: np.ndarray,
    title: str = "Encoder x LLM CKA Heatmap",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """CKA heatmap for encoder x LLM combinations.

    Args:
        encoder_names: Row labels
        llm_names: Column labels
        scores: 2D array [n_encoders, n_llms]
    """
    fig, ax = create_figure(figsize=(max(8, len(llm_names) * 2), max(5, len(encoder_names) * 1.5)))
    sns.heatmap(
        scores, annot=True, fmt=".4f", cmap="YlOrRd",
        xticklabels=[n.upper() for n in llm_names],
        yticklabels=[n.upper() for n in encoder_names],
        ax=ax, linewidths=1, linecolor="black",
    )
    style_axis(ax, title=title, grid=False)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_elas_results(
    elas_scores: list,
    title: str = "ELAS Score Matrix",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize ELAS scores for encoder x LLM combinations.

    Args:
        elas_scores: List of ELASScore dataclass instances
    """
    encoders = sorted(set(s.encoder for s in elas_scores))
    llms = sorted(set(s.llm for s in elas_scores))
    matrix = np.zeros((len(encoders), len(llms)))

    for s in elas_scores:
        i = encoders.index(s.encoder)
        j = llms.index(s.llm)
        matrix[i, j] = s.score

    return plot_cka_heatmap(encoders, llms, matrix, title=title, output_path=output_path)


def plot_alignment_summary(
    metrics: Dict,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Summary dashboard with CKA, MSE, and cosine metrics.

    Args:
        metrics: {encoder_name: AlignmentMetrics}
    """
    fig, axes = create_figure(1, 3, figsize=(18, 5))

    names = list(metrics.keys())
    colors = [get_model_color(n) for n in names]

    # CKA Linear
    values = [m.cka_linear for m in metrics.values()]
    axes[0].bar(names, values, color=colors, edgecolor="black")
    for x_pos, val in enumerate(values):
        axes[0].text(x_pos, val + 0.005, f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    style_axis(axes[0], title="CKA (Linear)", ylabel="Score")

    # Projection MSE
    values = [m.projection_mse for m in metrics.values()]
    axes[1].bar(names, values, color=colors, edgecolor="black")
    for x_pos, val in enumerate(values):
        axes[1].text(x_pos, val + 0.001, f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    style_axis(axes[1], title="Projection MSE", ylabel="MSE")

    # Cosine after projection
    values = [m.cosine_after_proj for m in metrics.values()]
    axes[2].bar(names, values, color=colors, edgecolor="black")
    for x_pos, val in enumerate(values):
        axes[2].text(x_pos, val + 0.005, f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    style_axis(axes[2], title="Cosine Similarity (after proj)", ylabel="Cosine")

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig

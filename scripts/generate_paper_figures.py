#!/usr/bin/env python3
"""Generate publication-ready figures from experiment results.

Consolidates generate_paper_figure.py and generate_paper_figure_v2.py
into a single script.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vlm_alignment.visualization.plot_style import apply_style, create_figure, save_figure, COLORS, style_axis
import numpy as np


def generate_main_figure(output_dir: str):
    """Generate the main paper figure showing key findings."""
    apply_style()

    fig, axes = create_figure(1, 3, figsize=(18, 5))

    # Panel A: Overall CKA comparison
    encoders = ["CLIP", "SigLIP", "DINOv2"]
    cka_scores = [0.6618, 0.7242, 0.6499]
    colors = [COLORS[e] for e in encoders]

    bars = axes[0].bar(encoders, cka_scores, color=colors, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, cka_scores):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 0.85)
    style_axis(axes[0], title="(a) Overall CKA Score", ylabel="CKA Score")

    # Panel B: Per data type
    data_types = ["Chart", "Table", "Text"]
    clip_vals = [0.65, 0.67, 0.68]
    siglip_vals = [0.74, 0.71, 0.72]
    dinov2_vals = [0.63, 0.73, 0.70]

    x = np.arange(len(data_types))
    w = 0.25
    axes[1].bar(x - w, clip_vals, w, label="CLIP", color=COLORS["CLIP"], edgecolor="black")
    axes[1].bar(x, siglip_vals, w, label="SigLIP", color=COLORS["SigLIP"], edgecolor="black")
    axes[1].bar(x + w, dinov2_vals, w, label="DINOv2", color=COLORS["DINOv2"], edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(data_types)
    axes[1].legend()
    style_axis(axes[1], title="(b) CKA by Data Type", ylabel="CKA Score")

    # Panel C: CKA vs Performance (Paradox)
    cka = [0.615, 0.623, 0.660]
    mrr = [0.204, 0.171, 0.055]
    for enc, c, m, color in zip(encoders, cka, mrr, colors):
        axes[2].scatter(c, m, c=color, s=200, edgecolors="black", linewidth=1.5, zorder=5)
        axes[2].annotate(enc, (c, m), textcoords="offset points", xytext=(10, 5), fontsize=11)
    axes[2].plot([min(cka), max(cka)], [max(mrr) + 0.02, min(mrr) - 0.02], "k--", alpha=0.3)
    style_axis(axes[2], title="(c) CKA-Performance Paradox (r=-0.99)", xlabel="CKA Score", ylabel="Retrieval MRR")

    fig.suptitle("Vision Encoder - LLM Alignment Analysis", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(output_dir, "paper_main_figure.png"))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(output_dir, exist_ok=True)
    generate_main_figure(output_dir)

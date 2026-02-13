"""CORAL analysis visualization.

Plots for Deep CORAL alignment analysis:
- Cross-modal CORAL distance comparison
- CKA vs CORAL scatter (paradox detection)
- Covariance matrix heatmaps
- Eigenvalue spectrum comparison
- EAS (Enhanced Alignment Score) dashboard
- Intra-modal similarity matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, style_axis, create_figure, save_figure,
)


def plot_coral_comparison(
    cross_modal_results: Dict,
    title: str = "CORAL Alignment: Vision Encoder vs LLM",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing CORAL distance and similarity across encoders.

    Args:
        cross_modal_results: {encoder_name: CrossModalResult}
    """
    fig, axes = create_figure(1, 2, figsize=(14, 5))

    names = list(cross_modal_results.keys())
    colors = [get_model_color(n) for n in names]

    # CORAL Distance (lower = better)
    distances = [r.coral.coral_distance for r in cross_modal_results.values()]
    bars = axes[0].bar(names, distances, color=colors, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, distances):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    style_axis(axes[0], title="CORAL Distance (lower = better aligned)",
               ylabel="CORAL Distance")

    # CORAL Similarity (higher = better)
    similarities = [r.coral.coral_similarity for r in cross_modal_results.values()]
    bars = axes[1].bar(names, similarities, color=colors, edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, similarities):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    style_axis(axes[1], title="CORAL Similarity (higher = better aligned)",
               ylabel="Similarity Score")
    axes[1].set_ylim(0, 1.15)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_cka_vs_coral(
    cross_modal_results: Dict,
    title: str = "CKA vs CORAL: Metric Comparison",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter plot showing CKA vs CORAL for each encoder.

    When points deviate from the diagonal, CKA and CORAL disagree -
    this reveals the CKA-Performance Paradox.

    Args:
        cross_modal_results: {encoder_name: CrossModalResult}
    """
    fig, ax = create_figure()

    names = list(cross_modal_results.keys())
    cka_vals = [r.cka_linear for r in cross_modal_results.values()]
    coral_vals = [r.coral.coral_similarity for r in cross_modal_results.values()]
    colors = [get_model_color(n) for n in names]

    ax.scatter(cka_vals, coral_vals, c=colors, s=200, edgecolors="black",
               linewidth=2, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name.upper(), (cka_vals[i], coral_vals[i]),
                    textcoords="offset points", xytext=(10, 5),
                    fontsize=12, fontweight="bold")

    # Diagonal reference
    lims = [
        min(min(cka_vals, default=0), min(coral_vals, default=0)) - 0.05,
        max(max(cka_vals, default=1), max(coral_vals, default=1)) + 0.05,
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="y = x")
    ax.legend()

    style_axis(ax, title=title, xlabel="CKA Score", ylabel="CORAL Similarity")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_covariance_heatmaps(
    cross_modal_results: Dict,
    max_dim: int = 50,
    title: str = "Covariance Matrix Comparison",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmaps of covariance matrices for vision and text embeddings.

    Shows the top-left corner (max_dim x max_dim) of each covariance matrix.

    Args:
        cross_modal_results: {encoder_name: CrossModalResult}
        max_dim: Maximum dimensions to show
    """
    names = list(cross_modal_results.keys())
    n = len(names)

    fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))
    apply_style()

    if n == 1:
        axes = [axes]

    for i, (enc_name, result) in enumerate(cross_modal_results.items()):
        cov_v = result.coral.covariance_source[:max_dim, :max_dim]
        cov_t = result.coral.covariance_target[:max_dim, :max_dim]
        cov_diff = cov_v - cov_t

        sns.heatmap(cov_v, ax=axes[i][0], cmap="RdBu_r", center=0,
                    xticklabels=False, yticklabels=False)
        axes[i][0].set_title(f"{enc_name.upper()} - Vision Cov", fontweight="bold")

        sns.heatmap(cov_t, ax=axes[i][1], cmap="RdBu_r", center=0,
                    xticklabels=False, yticklabels=False)
        axes[i][1].set_title(f"{enc_name.upper()} - Text Cov", fontweight="bold")

        sns.heatmap(cov_diff, ax=axes[i][2], cmap="RdBu_r", center=0,
                    xticklabels=False, yticklabels=False)
        axes[i][2].set_title(f"{enc_name.upper()} - Difference", fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_eigenvalue_spectrum(
    cross_modal_results: Dict,
    top_k: int = 30,
    title: str = "Eigenvalue Spectrum Comparison",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare eigenvalue spectra between vision and text covariances.

    Similar spectra indicate similar representational geometry.

    Args:
        cross_modal_results: {encoder_name: CrossModalResult}
        top_k: Number of top eigenvalues to show
    """
    names = list(cross_modal_results.keys())
    n = len(names)

    fig, axes = create_figure(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (enc_name, result) in enumerate(cross_modal_results.items()):
        eig_v = result.coral.eigenvalues_source[:top_k]
        eig_t = result.coral.eigenvalues_target[:top_k]

        x = np.arange(len(eig_v))
        axes[i].plot(x, eig_v, "o-", label="Vision", color=get_model_color(enc_name),
                     linewidth=2, markersize=5)
        axes[i].plot(x[:len(eig_t)], eig_t, "s--", label="Text", color="#666666",
                     linewidth=2, markersize=5)

        spec_div = result.coral.spectral_divergence
        axes[i].text(0.95, 0.95, f"Spec.Div={spec_div:.3f}",
                     transform=axes[i].transAxes, ha="right", va="top",
                     fontsize=10, bbox=dict(boxstyle="round", facecolor="lightyellow",
                                            edgecolor="orange"))

        style_axis(axes[i], title=f"{enc_name.upper()}", xlabel="Component",
                   ylabel="Eigenvalue")
        axes[i].legend()
        axes[i].set_yscale("log")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_eas_dashboard(
    eas_results: Dict,
    title: str = "Enhanced Alignment Score (EAS) Dashboard",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Comprehensive EAS visualization with component breakdown.

    Shows CKA, CORAL, Discriminability components and final EAS.

    Args:
        eas_results: {encoder_name: EASResult}
    """
    fig, axes = create_figure(1, 3, figsize=(18, 6))

    names = list(eas_results.keys())
    colors = [get_model_color(n) for n in names]
    x = np.arange(len(names))
    width = 0.25

    # Plot 1: Component scores
    cka_vals = [r.cka for r in eas_results.values()]
    coral_vals = [r.coral_score for r in eas_results.values()]
    disc_vals = [r.discriminability for r in eas_results.values()]

    axes[0].bar(x - width, cka_vals, width, label="CKA", alpha=0.85,
                color="#3b7faf", edgecolor="black")
    axes[0].bar(x, coral_vals, width, label="CORAL", alpha=0.85,
                color="#ff913d", edgecolor="black")
    axes[0].bar(x + width, disc_vals, width, label="Discriminability", alpha=0.85,
                color="#4ba0b1", edgecolor="black")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.upper() for n in names])
    axes[0].legend()
    axes[0].set_ylim(0, 1.15)
    style_axis(axes[0], title="(a) Component Scores", ylabel="Score")

    # Plot 2: CKA vs EAS scatter
    eas_vals = [r.eas for r in eas_results.values()]
    axes[1].scatter(cka_vals, eas_vals, c=colors, s=200, edgecolors="black", linewidth=2)
    for i, name in enumerate(names):
        axes[1].annotate(name.upper(), (cka_vals[i], eas_vals[i]),
                         textcoords="offset points", xytext=(10, 5), fontsize=11)

    lims = [
        min(min(cka_vals, default=0), min(eas_vals, default=0)) - 0.05,
        max(max(cka_vals, default=1), max(eas_vals, default=1)) + 0.05,
    ]
    axes[1].plot(lims, lims, "--", color="gray", alpha=0.5)
    style_axis(axes[1], title="(b) CKA vs EAS", xlabel="CKA", ylabel="EAS")

    # Plot 3: Ranking table
    axes[2].axis("off")
    cka_rank = sorted(names, key=lambda n: eas_results[n].cka, reverse=True)
    eas_rank = sorted(names, key=lambda n: eas_results[n].eas, reverse=True)
    disc_rank = sorted(names, key=lambda n: eas_results[n].discriminability, reverse=True)

    lines = [
        "Ranking Comparison",
        "=" * 35,
        "",
        "By CKA (structural):",
    ]
    for i, n in enumerate(cka_rank, 1):
        lines.append(f"  {i}. {n.upper()} ({eas_results[n].cka:.3f})")
    lines.extend(["", "By Discriminability:"])
    for i, n in enumerate(disc_rank, 1):
        lines.append(f"  {i}. {n.upper()} ({eas_results[n].discriminability:.3f})")
    lines.extend(["", "By EAS (balanced):"])
    for i, n in enumerate(eas_rank, 1):
        lines.append(f"  {i}. {n.upper()} ({eas_results[n].eas:.3f})")
    lines.extend(["", "=" * 35, "EAS = 0.3*CKA + 0.3*CORAL + 0.4*Disc"])

    text = "\n".join(lines)
    axes[2].text(0.05, 0.95, text, transform=axes[2].transAxes, fontsize=10,
                 fontfamily="monospace", verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"))
    style_axis(axes[2], title="(c) Ranking", grid=False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_intra_modal_matrix(
    intra_results: List,
    title: str = "Intra-Modal CORAL Similarity",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of pairwise CORAL similarity within a modality.

    Args:
        intra_results: List of IntraModalResult
    """
    if not intra_results:
        fig, ax = create_figure()
        ax.text(0.5, 0.5, "No pairwise comparisons", ha="center", va="center")
        return fig

    # Collect unique names
    all_names = set()
    for r in intra_results:
        all_names.add(r.name_a)
        all_names.add(r.name_b)
    names = sorted(all_names)
    n = len(names)

    # Build matrix
    matrix = np.eye(n)  # diagonal = 1 (self-similarity)
    for r in intra_results:
        i = names.index(r.name_a)
        j = names.index(r.name_b)
        matrix[i, j] = r.coral.coral_similarity
        matrix[j, i] = r.coral.coral_similarity

    fig, ax = create_figure(figsize=(max(6, n * 2), max(5, n * 1.5)))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=[n.upper() for n in names],
        yticklabels=[n.upper() for n in names],
        ax=ax, linewidths=1, linecolor="black", vmin=0, vmax=1,
    )
    style_axis(ax, title=title, grid=False)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_coral_full_dashboard(
    analysis_results: Dict,
    output_dir: str = "outputs",
) -> Dict[str, plt.Figure]:
    """Generate all CORAL analysis plots.

    Args:
        analysis_results: Output from CORALAnalyzer.full_analysis()
        output_dir: Directory to save plots

    Returns:
        Dict of {plot_name: figure}
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    if "cross_modal" in analysis_results:
        cross = analysis_results["cross_modal"]

        fig = plot_coral_comparison(
            cross, output_path=os.path.join(output_dir, "coral_comparison.png"))
        figures["coral_comparison"] = fig

        fig = plot_cka_vs_coral(
            cross, output_path=os.path.join(output_dir, "cka_vs_coral.png"))
        figures["cka_vs_coral"] = fig

        fig = plot_covariance_heatmaps(
            cross, output_path=os.path.join(output_dir, "covariance_heatmaps.png"))
        figures["covariance_heatmaps"] = fig

        fig = plot_eigenvalue_spectrum(
            cross, output_path=os.path.join(output_dir, "eigenvalue_spectrum.png"))
        figures["eigenvalue_spectrum"] = fig

    if "intra_modal" in analysis_results:
        fig = plot_intra_modal_matrix(
            analysis_results["intra_modal"],
            output_path=os.path.join(output_dir, "intra_modal_similarity.png"))
        figures["intra_modal"] = fig

    if "eas" in analysis_results:
        fig = plot_eas_dashboard(
            analysis_results["eas"],
            output_path=os.path.join(output_dir, "eas_dashboard.png"))
        figures["eas_dashboard"] = fig

    return figures

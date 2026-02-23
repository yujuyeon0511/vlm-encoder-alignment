#!/usr/bin/env python3
"""Generate all publication-ready figures for the VLM encoder alignment paper.

Usage:
    # Full pipeline: extract embeddings + generate all figures
    python scripts/generate_paper_figures.py --device cuda

    # Quick re-render from cached embeddings (no model loading)
    python scripts/generate_paper_figures.py --skip-extraction

    # Generate only specific figures
    python scripts/generate_paper_figures.py --figures 1 6 --skip-extraction

    # Use more samples
    python scripts/generate_paper_figures.py --n-samples 50 --device cuda
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, get_data_type_color,
    style_axis, create_figure, save_figure, COLORS,
)
from vlm_alignment.visualization.paper_utils import (
    add_panel_label, bar_chart_on_axis, grouped_bar_on_axis, scatter_with_labels,
)
from vlm_alignment.visualization.embedding_space import (
    reduce_tsne, reduce_umap, procrustes_align,
)

# ==============================================================================
# Experiment results from Phase 2 runs (extracted from log files)
# ==============================================================================
EXPERIMENT_RESULTS = {
    "encoders": ["CLIP", "SigLIP", "DINOv2"],
    "llm": "Qwen2.5-7B",

    # Cross-modal CKA (from CORAL analysis, PCA-aligned dimensions)
    "cka_overall": {"CLIP": 0.9902, "SigLIP": 0.9800, "DINOv2": 0.9148},

    # Per data-type CKA (from compare experiment, raw vision vs text)
    "cka_by_type": {
        "chart": {"CLIP": 0.0611, "SigLIP": 0.0501, "DINOv2": 0.0177},
        "table": {"CLIP": 0.0943, "SigLIP": 0.0703, "DINOv2": 0.0150},
        "text":  {"CLIP": 0.0000, "SigLIP": 0.0829, "DINOv2": 0.0000},
    },

    # CORAL metrics
    "coral_similarity": {"CLIP": 1.0000, "SigLIP": 1.0000, "DINOv2": 1.0000},
    "spectral_divergence": {"CLIP": 0.1238, "SigLIP": 0.3687, "DINOv2": 0.1330},

    # EAS components
    "eas": {
        "CLIP":   {"cka": 0.9902, "coral": 1.0000, "disc": 0.9439, "eas": 0.9746},
        "SigLIP": {"cka": 0.9800, "coral": 1.0000, "disc": 0.7846, "eas": 0.9079},
        "DINOv2": {"cka": 0.9148, "coral": 1.0000, "disc": 0.9867, "eas": 0.9691},
    },

    # E2E retrieval performance
    "e2e": {
        "CLIP":   {"cka": 0.9902, "recall_1": 0.027, "recall_5": 0.127, "mrr": 0.096},
        "SigLIP": {"cka": 0.9800, "recall_1": 0.047, "recall_5": 0.160, "mrr": 0.126},
        "DINOv2": {"cka": 0.9148, "recall_1": 0.040, "recall_5": 0.140, "mrr": 0.109},
    },
    "cka_mrr_correlation": -0.0475,

    # Intra-modal similarity
    "intra_modal": {
        ("CLIP", "SigLIP"): {"cka": 0.9922, "coral": 1.0000},
        ("CLIP", "DINOv2"): {"cka": 0.9547, "coral": 1.0000},
        ("SigLIP", "DINOv2"): {"cka": 0.9563, "coral": 1.0000},
    },

    # Pretrained CKA baseline (VLM internal vs open encoder)
    "pretrained_cka": {
        "Qwen2.5-VL\n(internal)": 0.0229,
        "LLaVA-OV\n(internal)": 0.0672,
        "CLIP\n(open)": 0.9902,
        "SigLIP\n(open)": 0.9800,
        "DINOv2\n(open)": 0.9148,
    },
}


# ==============================================================================
# Embedding Cache
# ==============================================================================
class EmbeddingCache:
    """Save and load embeddings to avoid re-extraction."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, vision_embs: Dict[str, np.ndarray], text_embs: np.ndarray,
             labels: List[str], metadata: Optional[dict] = None):
        for name, emb in vision_embs.items():
            np.save(self.cache_dir / f"vision_{name}.npy", emb)
        np.save(self.cache_dir / "text_embeddings.npy", text_embs)
        np.save(self.cache_dir / "labels.npy", np.array(labels))
        if metadata:
            with open(self.cache_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        print(f"  Cached embeddings to {self.cache_dir}")

    def load(self, encoder_names: List[str]):
        vision_embs = {}
        for name in encoder_names:
            path = self.cache_dir / f"vision_{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Cache not found: {path}")
            vision_embs[name] = np.load(path)
        text_embs = np.load(self.cache_dir / "text_embeddings.npy")
        labels = np.load(self.cache_dir / "labels.npy").tolist()
        return vision_embs, text_embs, labels

    def exists(self, encoder_names: List[str]) -> bool:
        for name in encoder_names:
            if not (self.cache_dir / f"vision_{name}.npy").exists():
                return False
        return (
            (self.cache_dir / "text_embeddings.npy").exists()
            and (self.cache_dir / "labels.npy").exists()
        )


# ==============================================================================
# Data Loading & Embedding Extraction
# ==============================================================================
def extract_all_embeddings(
    encoder_names: List[str],
    llm_name: str,
    n_per_type: int,
    device: str,
    cache: EmbeddingCache,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    """Load data and extract embeddings from all encoders + LLM."""
    import torch
    from vlm_alignment.data.dataset import VLMDataset
    from vlm_alignment.models.vision_encoders import VisionEncoderManager
    from vlm_alignment.models.llm_loaders import LLMManager

    print(f"\n  Loading dataset ({n_per_type} samples per type)...")
    dataset = VLMDataset()
    samples, labels = dataset.load_mixed(
        n_per_type=n_per_type, data_types=["chart", "table", "text"]
    )
    images, texts = dataset.get_images_and_texts(samples)
    print(f"  Loaded {len(images)} samples")

    print(f"\n  Extracting vision embeddings...")
    ve_mgr = VisionEncoderManager(device=device)
    vision_embs = ve_mgr.extract_multi_encoder(encoder_names, images)
    ve_mgr.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  Extracting text embeddings ({llm_name})...")
    llm_mgr = LLMManager(device=device)
    text_embs = llm_mgr.extract_text_embeddings(llm_name, texts)
    llm_mgr.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cache.save(vision_embs, text_embs, labels, metadata={
        "encoders": encoder_names, "llm": llm_name,
        "n_per_type": n_per_type, "n_total": len(images),
    })

    return vision_embs, text_embs, labels


def compute_analysis_results(
    vision_embs: Dict[str, np.ndarray],
    text_embs: np.ndarray,
    labels: List[str],
    llm_name: str,
) -> Dict:
    """Run CORALAnalyzer.full_analysis()."""
    from vlm_alignment.analysis.coral import CORALAnalyzer
    analyzer = CORALAnalyzer()
    return analyzer.full_analysis(
        vision_embeddings=vision_embs,
        text_embeddings=text_embs,
        llm_name=llm_name,
        labels=np.array(labels),
    )


# ==============================================================================
# Figure 1: CKA Alignment Overview
# ==============================================================================
def generate_figure_1(output_dir: str, **kwargs) -> plt.Figure:
    """Figure 1: CKA Alignment Overview.

    (a) Overall CKA scores  (b) CKA by data type  (c) CKA-Performance Paradox
    """
    R = EXPERIMENT_RESULTS
    encoders = R["encoders"]

    fig, axes = create_figure(1, 3, figsize=(18, 5.5))

    # (a) Overall CKA bar chart
    cka_vals = [R["cka_overall"][e] for e in encoders]
    bar_chart_on_axis(axes[0], encoders, cka_vals,
                      title="Overall CKA Score", ylabel="CKA Score")
    add_panel_label(axes[0], "(a)")

    # (b) CKA by data type
    series = {e.lower(): [R["cka_by_type"][dt][e] for dt in ["chart", "table", "text"]]
              for e in encoders}
    grouped_bar_on_axis(axes[1], ["chart", "table", "text"], series,
                        title="CKA by Data Type", ylabel="CKA Score", fmt=".4f")
    add_panel_label(axes[1], "(b)")

    # (c) CKA vs MRR scatter (Paradox)
    cka_vals = [R["e2e"][e]["cka"] for e in encoders]
    mrr_vals = [R["e2e"][e]["mrr"] for e in encoders]
    scatter_with_labels(
        axes[2], cka_vals, mrr_vals, encoders,
        title="CKA-Performance Paradox",
        xlabel="CKA Score", ylabel="Retrieval MRR",
        show_trend=True,
        annotation_text=f"Pearson r = {R['cka_mrr_correlation']:.4f}",
    )
    add_panel_label(axes[2], "(c)")

    fig.suptitle("Figure 1: Vision Encoder – LLM Alignment Analysis",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ==============================================================================
# Figure 2: Deep CORAL Analysis
# ==============================================================================
def generate_figure_2(output_dir: str, cross_modal_results=None, **kwargs) -> plt.Figure:
    """Figure 2: Deep CORAL Analysis.

    (a) Spectral divergence  (b) CKA vs Spectral Div  (c) Eigenvalue spectrum
    """
    R = EXPERIMENT_RESULTS
    encoders = R["encoders"]

    fig, axes = create_figure(1, 3, figsize=(18, 5.5))

    # (a) Spectral Divergence bar chart
    spec_vals = [R["spectral_divergence"][e] for e in encoders]
    bar_chart_on_axis(axes[0], encoders, spec_vals,
                      title="Spectral Divergence", ylabel="Divergence (lower = better)")
    add_panel_label(axes[0], "(a)")

    # (b) CKA vs Spectral Divergence scatter
    cka_vals = [R["cka_overall"][e] for e in encoders]
    scatter_with_labels(
        axes[1], cka_vals, spec_vals, encoders,
        title="CKA vs Spectral Divergence",
        xlabel="CKA Score", ylabel="Spectral Divergence",
        annotation_text="Higher CKA ≠ Lower Divergence",
    )
    add_panel_label(axes[1], "(b)")

    # (c) Eigenvalue spectrum (from live data or placeholder)
    if cross_modal_results:
        for enc_name, result in cross_modal_results.items():
            eig_v = result.coral.eigenvalues_source[:30]
            eig_t = result.coral.eigenvalues_target[:30]
            x = np.arange(len(eig_v))
            axes[2].plot(x, eig_v, "o-", label=f"{enc_name} (Vision)",
                         color=get_model_color(enc_name), linewidth=2, markersize=4)
        # Plot text eigenvalues (same for all since same LLM)
        last_result = list(cross_modal_results.values())[-1]
        eig_t = last_result.coral.eigenvalues_target[:30]
        axes[2].plot(np.arange(len(eig_t)), eig_t, "s--", label="Text (LLM)",
                     color="#666666", linewidth=2, markersize=4)
        axes[2].set_yscale("log")
        axes[2].legend(fontsize=9)
        style_axis(axes[2], title="Eigenvalue Spectrum",
                   xlabel="Component", ylabel="Eigenvalue")
    else:
        # Placeholder when no live data
        axes[2].text(0.5, 0.5, "Eigenvalue Spectrum\n(requires --device cuda)",
                     ha="center", va="center", fontsize=12, transform=axes[2].transAxes,
                     bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="orange"))
        style_axis(axes[2], title="Eigenvalue Spectrum", grid=False)
    add_panel_label(axes[2], "(c)")

    fig.suptitle("Figure 2: Deep CORAL Distribution Analysis",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ==============================================================================
# Figure 3: EAS Dashboard
# ==============================================================================
def generate_figure_3(output_dir: str, eas_results=None, **kwargs) -> plt.Figure:
    """Figure 3: Enhanced Alignment Score (EAS).

    (a) Component scores  (b) CKA vs EAS scatter  (c) Ranking comparison
    """
    R = EXPERIMENT_RESULTS
    encoders = R["encoders"]

    # Use live or hardcoded EAS data
    if eas_results:
        cka_vals = [eas_results[e.lower()].cka for e in encoders]
        coral_vals = [eas_results[e.lower()].coral_score for e in encoders]
        disc_vals = [eas_results[e.lower()].discriminability for e in encoders]
        eas_vals = [eas_results[e.lower()].eas for e in encoders]
    else:
        cka_vals = [R["eas"][e]["cka"] for e in encoders]
        coral_vals = [R["eas"][e]["coral"] for e in encoders]
        disc_vals = [R["eas"][e]["disc"] for e in encoders]
        eas_vals = [R["eas"][e]["eas"] for e in encoders]

    fig, axes = create_figure(1, 3, figsize=(18, 6))

    # (a) Component scores grouped bar
    x = np.arange(len(encoders))
    w = 0.25
    axes[0].bar(x - w, cka_vals, w, label="CKA", alpha=0.85,
                color="#3b7faf", edgecolor="black")
    axes[0].bar(x, coral_vals, w, label="CORAL", alpha=0.85,
                color="#ff913d", edgecolor="black")
    axes[0].bar(x + w, disc_vals, w, label="Discriminability", alpha=0.85,
                color="#4ba0b1", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(encoders)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 1.15)
    style_axis(axes[0], title="Component Scores", ylabel="Score")
    add_panel_label(axes[0], "(a)")

    # (b) CKA vs EAS scatter
    scatter_with_labels(
        axes[1], cka_vals, eas_vals, encoders,
        title="CKA vs EAS",
        xlabel="CKA Score", ylabel="EAS Score",
        show_diagonal=True,
    )
    add_panel_label(axes[1], "(b)")

    # (c) Ranking table
    axes[2].axis("off")
    cka_rank = sorted(range(len(encoders)), key=lambda i: -cka_vals[i])
    eas_rank = sorted(range(len(encoders)), key=lambda i: -eas_vals[i])
    disc_rank = sorted(range(len(encoders)), key=lambda i: -disc_vals[i])

    lines = [
        "       Ranking Comparison",
        "  " + "=" * 38,
        "",
        "  By CKA (structural):",
    ]
    for rank, i in enumerate(cka_rank, 1):
        lines.append(f"    {rank}. {encoders[i]:>7s}  ({cka_vals[i]:.4f})")
    lines.extend(["", "  By Discriminability:"])
    for rank, i in enumerate(disc_rank, 1):
        lines.append(f"    {rank}. {encoders[i]:>7s}  ({disc_vals[i]:.4f})")
    lines.extend(["", "  By EAS (balanced):"])
    for rank, i in enumerate(eas_rank, 1):
        lines.append(f"    {rank}. {encoders[i]:>7s}  ({eas_vals[i]:.4f})")
    lines.extend(["", "  " + "=" * 38,
                   "  EAS = 0.3·CKA + 0.3·CORAL + 0.4·Disc"])

    axes[2].text(0.05, 0.95, "\n".join(lines), transform=axes[2].transAxes,
                 fontsize=11, fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round", facecolor="lightyellow",
                           edgecolor="orange", alpha=0.9))
    style_axis(axes[2], title="Ranking", grid=False)
    add_panel_label(axes[2], "(c)")

    fig.suptitle("Figure 3: Enhanced Alignment Score (EAS) Dashboard",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ==============================================================================
# Figure 4: Embedding Space Visualization
# ==============================================================================
def generate_figure_4(
    output_dir: str,
    vision_embeddings: Optional[Dict[str, np.ndarray]] = None,
    text_embeddings: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    **kwargs,
) -> plt.Figure:
    """Figure 4: Embedding Space Visualization.

    Row 1: t-SNE per encoder colored by data type
    Row 2: Vision + Text overlay per encoder
    """
    if vision_embeddings is None or text_embeddings is None:
        raise ValueError("Figure 4 requires live embeddings (cannot use --skip-extraction)")

    from vlm_alignment.analysis.coral import align_dimensions

    encoder_names = list(vision_embeddings.keys())
    n_enc = len(encoder_names)

    fig, axes = plt.subplots(2, n_enc, figsize=(6 * n_enc, 11))
    apply_style()

    if n_enc == 1:
        axes = axes.reshape(2, 1)

    # Row 1: t-SNE colored by data type
    unique_labels = sorted(set(labels)) if labels else []
    for j, (name, v_emb) in enumerate(vision_embeddings.items()):
        reduced = reduce_tsne(v_emb)

        if labels and unique_labels:
            for lbl in unique_labels:
                mask = np.array([l == lbl for l in labels])
                axes[0][j].scatter(
                    reduced[mask, 0], reduced[mask, 1],
                    c=get_data_type_color(lbl), label=lbl.capitalize(),
                    s=40, alpha=0.7, edgecolors="white", linewidth=0.5,
                )
            if j == 0:
                axes[0][j].legend(fontsize=9, loc="upper left")
        else:
            axes[0][j].scatter(reduced[:, 0], reduced[:, 1],
                               c=get_model_color(name), s=40, alpha=0.7)

        style_axis(axes[0][j], title=f"{name.upper()} (t-SNE)", grid=False)
        add_panel_label(axes[0][j], f"({chr(97 + j)})")

    # Row 2: Vision + Text overlay
    for j, (name, v_emb) in enumerate(vision_embeddings.items()):
        n = min(v_emb.shape[0], text_embeddings.shape[0])
        v_proj, t_proj = align_dimensions(v_emb[:n], text_embeddings[:n])
        combined = np.vstack([v_proj, t_proj])
        combined_2d = reduce_tsne(combined)
        mid = len(v_proj)

        axes[1][j].scatter(
            combined_2d[:mid, 0], combined_2d[:mid, 1],
            c=get_model_color(name), label="Vision", s=35, alpha=0.6,
            edgecolors="white", linewidth=0.5,
        )
        axes[1][j].scatter(
            combined_2d[mid:, 0], combined_2d[mid:, 1],
            c="#666666", label="Text (LLM)", s=35, alpha=0.6,
            marker="x", linewidth=1.5,
        )
        if j == 0:
            axes[1][j].legend(fontsize=9, loc="upper left")

        style_axis(axes[1][j],
                   title=f"{name.upper()} + {EXPERIMENT_RESULTS['llm']} Overlay",
                   grid=False)
        add_panel_label(axes[1][j], f"({chr(100 + j)})")

    fig.suptitle("Figure 4: Embedding Space Visualization",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ==============================================================================
# Figure 5: Representational Geometry
# ==============================================================================
def generate_figure_5(
    output_dir: str,
    vision_embeddings: Optional[Dict[str, np.ndarray]] = None,
    text_embeddings: Optional[np.ndarray] = None,
    **kwargs,
) -> plt.Figure:
    """Figure 5: Representational Geometry.

    Row 1: Covariance heatmaps (vision covariance per encoder)
    Row 2: Procrustes alignment before/after
    """
    if vision_embeddings is None or text_embeddings is None:
        raise ValueError("Figure 5 requires live embeddings (cannot use --skip-extraction)")

    from vlm_alignment.analysis.coral import align_dimensions, compute_covariance

    encoder_names = list(vision_embeddings.keys())
    n_enc = len(encoder_names)

    fig, axes = plt.subplots(2, n_enc, figsize=(6 * n_enc, 11))
    apply_style()

    if n_enc == 1:
        axes = axes.reshape(2, 1)

    max_dim = 50

    # Row 1: Covariance heatmaps
    for j, (name, v_emb) in enumerate(vision_embeddings.items()):
        n = min(v_emb.shape[0], text_embeddings.shape[0])
        v_proj, t_proj = align_dimensions(v_emb[:n], text_embeddings[:n])
        cov_v = compute_covariance(v_proj)[:max_dim, :max_dim]
        cov_t = compute_covariance(t_proj)[:max_dim, :max_dim]
        cov_diff = cov_v - cov_t

        sns.heatmap(cov_diff, ax=axes[0][j], cmap="RdBu_r", center=0,
                    xticklabels=False, yticklabels=False, cbar_kws={"shrink": 0.8})
        style_axis(axes[0][j],
                   title=f"{name.upper()} Cov. Difference (V−T)",
                   grid=False)
        add_panel_label(axes[0][j], f"({chr(97 + j)})")

    # Row 2: Procrustes alignment
    # Use first encoder as reference target
    target_name = encoder_names[0]
    target_emb = vision_embeddings[target_name]

    for j, name in enumerate(encoder_names):
        src_emb = vision_embeddings[name]

        if name == target_name:
            # Self: show original t-SNE
            reduced = reduce_tsne(src_emb)
            axes[1][j].scatter(reduced[:, 0], reduced[:, 1],
                               c=get_model_color(name), s=30, alpha=0.7,
                               edgecolors="white", linewidth=0.5)
            style_axis(axes[1][j],
                       title=f"{name.upper()} (reference)", grid=False)
        else:
            # Procrustes align source to target, show combined t-SNE
            aligned, error = procrustes_align(src_emb, target_emb)
            tgt_centered = target_emb[:, :aligned.shape[1]] - \
                target_emb[:, :aligned.shape[1]].mean(axis=0)
            combined = np.vstack([aligned, tgt_centered])
            combined_2d = reduce_tsne(combined)
            mid = len(aligned)

            axes[1][j].scatter(
                combined_2d[:mid, 0], combined_2d[:mid, 1],
                c=get_model_color(name), label=f"{name.upper()} (aligned)",
                s=30, alpha=0.6, edgecolors="white", linewidth=0.5,
            )
            axes[1][j].scatter(
                combined_2d[mid:, 0], combined_2d[mid:, 1],
                c=get_model_color(target_name),
                label=f"{target_name.upper()} (target)",
                s=30, alpha=0.6, marker="x", linewidth=1.5,
            )
            axes[1][j].legend(fontsize=8)
            style_axis(axes[1][j],
                       title=f"Procrustes: {name.upper()} → {target_name.upper()} "
                             f"(err={error:.1f})",
                       grid=False)

        add_panel_label(axes[1][j], f"({chr(100 + j)})")

    fig.suptitle("Figure 5: Representational Geometry Analysis",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


# ==============================================================================
# Figure 6: Pretrained CKA Baseline Comparison
# ==============================================================================
def generate_figure_6(output_dir: str, **kwargs) -> plt.Figure:
    """Figure 6: Pretrained CKA Baseline.

    Compares VLM internal CKA (low) vs open encoder CKA (high),
    showing that pretrained encoders already have structural alignment.
    """
    R = EXPERIMENT_RESULTS

    # Load from JSON if available, else use hardcoded
    json_path = PROJECT_ROOT / "outputs" / "cka" / "pretrained_cka.json"
    if json_path.exists():
        with open(json_path) as f:
            pretrained = json.load(f)
        internal_data = {
            "Qwen2.5-VL\n(internal)": pretrained["qwen25vl"]["cka_linear"],
            "LLaVA-OV\n(internal)": pretrained["llava_ov"]["cka_linear"],
        }
    else:
        internal_data = {
            "Qwen2.5-VL\n(internal)": 0.0229,
            "LLaVA-OV\n(internal)": 0.0672,
        }

    open_data = {
        "CLIP\n(open)": R["cka_overall"]["CLIP"],
        "SigLIP\n(open)": R["cka_overall"]["SigLIP"],
        "DINOv2\n(open)": R["cka_overall"]["DINOv2"],
    }

    fig, axes = create_figure(1, 2, figsize=(14, 5.5))

    # (a) Side-by-side comparison
    all_names = list(internal_data.keys()) + list(open_data.keys())
    all_values = list(internal_data.values()) + list(open_data.values())
    colors = (
        ["#E74C3C", "#E74C3C"]  # Red for internal
        + [get_model_color("clip"), get_model_color("siglip"), get_model_color("dinov2")]
    )

    bars = axes[0].bar(range(len(all_names)), all_values, color=colors,
                       edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, all_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    axes[0].set_xticks(range(len(all_names)))
    axes[0].set_xticklabels(all_names, fontsize=10)
    axes[0].axvline(x=1.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].text(0.5, 0.95, "VLM Internal", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=9, color="#E74C3C", fontweight="bold")
    axes[0].text(0.7, 0.95, "Open Encoder", transform=axes[0].transAxes,
                 ha="left", va="top", fontsize=9, color="#3b7faf", fontweight="bold")
    style_axis(axes[0], title="CKA Score Comparison", ylabel="CKA (Linear)")
    add_panel_label(axes[0], "(a)")

    # (b) Ratio / gap visualization
    categories = ["VLM Internal\n(avg)", "Open Encoder\n(avg)"]
    avg_internal = np.mean(list(internal_data.values()))
    avg_open = np.mean(list(open_data.values()))
    ratio = avg_open / avg_internal if avg_internal > 0 else float("inf")

    bars = axes[1].bar(categories, [avg_internal, avg_open],
                       color=["#E74C3C", "#3b7faf"], edgecolor="black", linewidth=1.5)
    for bar, val in zip(bars, [avg_internal, avg_open]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f"{val:.4f}", ha="center", va="bottom",
                     fontsize=11, fontweight="bold")

    axes[1].annotate(
        f"{ratio:.1f}× gap", xy=(1, avg_open), xytext=(0.5, (avg_internal + avg_open) / 2),
        fontsize=14, fontweight="bold", ha="center",
        arrowprops=dict(arrowstyle="<->", color="black", lw=2),
    )
    style_axis(axes[1], title="Average CKA Gap", ylabel="CKA (Linear)")
    add_panel_label(axes[1], "(b)")

    fig.suptitle("Figure 6: Pretrained CKA Baseline – Internal vs Open Encoders",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ==============================================================================
# Orchestrator
# ==============================================================================
FIGURE_GENERATORS = {
    1: ("CKA Alignment Overview", generate_figure_1),
    2: ("Deep CORAL Analysis", generate_figure_2),
    3: ("EAS Dashboard", generate_figure_3),
    4: ("Embedding Space Visualization", generate_figure_4),
    5: ("Representational Geometry", generate_figure_5),
    6: ("Pretrained CKA Baseline", generate_figure_6),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda or cpu (default: auto)")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Samples per data type (default: 50)")
    parser.add_argument("--llm", type=str, default="qwen",
                        help="LLM for text embeddings (default: qwen)")
    parser.add_argument("--encoders", nargs="+", default=["clip", "siglip", "dinov2"])
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "paper_figures"))
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Use cached embeddings (no model loading)")
    parser.add_argument("--figures", nargs="+", type=int, default=None,
                        help="Specific figures to generate (e.g., --figures 1 2 4)")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"],
                        help="Output formats (default: png pdf)")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    cache = EmbeddingCache(os.path.join(args.output_dir, "cache"))

    figures_to_gen = args.figures or list(FIGURE_GENERATORS.keys())
    needs_embeddings = bool(set(figures_to_gen) & {4, 5})
    needs_analysis = bool(set(figures_to_gen) & {2, 3})

    print("=" * 60)
    print("Paper Figure Generation")
    print("=" * 60)
    print(f"  Figures: {figures_to_gen}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print(f"  Formats: {args.formats}")
    print("=" * 60)

    # Step 1: Get embeddings if needed
    vision_embs = text_embs = labels = None
    analysis_results = None

    if needs_embeddings or (needs_analysis and not args.skip_extraction):
        if args.skip_extraction:
            if cache.exists(args.encoders):
                print("\nLoading cached embeddings...")
                vision_embs, text_embs, labels = cache.load(args.encoders)
                print(f"  Loaded: {len(labels)} samples, "
                      f"encoders: {list(vision_embs.keys())}")
            else:
                if needs_embeddings:
                    print("\nERROR: --skip-extraction but no cache found.")
                    print(f"  Run without --skip-extraction first.")
                    sys.exit(1)
                else:
                    print("\n  No cache found, using hardcoded values for figures 2-3")
        else:
            print("\nExtracting embeddings...")
            vision_embs, text_embs, labels = extract_all_embeddings(
                encoder_names=args.encoders,
                llm_name=args.llm,
                n_per_type=args.n_samples,
                device=args.device,
                cache=cache,
            )

    # Step 2: Compute CORAL analysis if needed
    if needs_analysis and vision_embs is not None:
        print("\nComputing CORAL analysis...")
        analysis_results = compute_analysis_results(
            vision_embs, text_embs, labels, args.llm,
        )

    # Step 3: Generate figures
    cross_modal = analysis_results["cross_modal"] if analysis_results else None
    eas_res = analysis_results["eas"] if analysis_results else None

    generated = []
    for fig_num in figures_to_gen:
        title, gen_func = FIGURE_GENERATORS[fig_num]
        print(f"\n{'─' * 40}")
        print(f"  Figure {fig_num}: {title}")

        try:
            fig = gen_func(
                output_dir=args.output_dir,
                vision_embeddings=vision_embs,
                text_embeddings=text_embs,
                labels=labels,
                cross_modal_results=cross_modal,
                eas_results=eas_res,
            )

            save_path = os.path.join(args.output_dir, f"figure_{fig_num}.png")
            save_figure(fig, save_path, dpi=300, formats=args.formats)
            plt.close(fig)
            generated.append(fig_num)

        except Exception as e:
            print(f"  WARNING: Figure {fig_num} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Generated {len(generated)}/{len(figures_to_gen)} figures: {generated}")
    print(f"Output directory: {args.output_dir}")
    for fmt in args.formats:
        files = [f for f in os.listdir(args.output_dir) if f.endswith(f".{fmt}")]
        print(f"  .{fmt} files: {len(files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

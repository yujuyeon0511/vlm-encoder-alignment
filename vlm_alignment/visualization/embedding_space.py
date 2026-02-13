"""Embedding space visualization with t-SNE, UMAP, and Procrustes alignment."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

from vlm_alignment.visualization.plot_style import (
    apply_style, get_model_color, get_data_type_color,
    style_axis, create_figure, save_figure,
)

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def reduce_tsne(embeddings: np.ndarray, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    """Reduce to 2D using t-SNE."""
    n = embeddings.shape[0]
    perp = min(perplexity, max(5, n - 1))
    return TSNE(n_components=2, perplexity=perp, random_state=seed).fit_transform(embeddings)


def reduce_umap(embeddings: np.ndarray, n_neighbors: int = 15, seed: int = 42) -> np.ndarray:
    """Reduce to 2D using UMAP."""
    if not UMAP_AVAILABLE:
        return reduce_tsne(embeddings)
    n = embeddings.shape[0]
    nn = min(n_neighbors, max(2, n - 1))
    return umap.UMAP(n_neighbors=nn, random_state=seed).fit_transform(embeddings)


def procrustes_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Align source embedding space to target via Procrustes analysis.

    Handles different dimensions by projecting to common space via PCA.

    Returns:
        (aligned_source, alignment_error)
    """
    from sklearn.decomposition import PCA

    s_dim, t_dim = source.shape[1], target.shape[1]
    n = source.shape[0]

    if s_dim != t_dim:
        min_dim = min(s_dim, t_dim, n - 1)
        source = PCA(n_components=min_dim).fit_transform(source)
        target = PCA(n_components=min_dim).fit_transform(target)

    src_c = source - source.mean(axis=0)
    tgt_c = target - target.mean(axis=0)

    M = tgt_c.T @ src_c
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt

    aligned = src_c @ R.T
    error = np.sqrt(np.sum((aligned - tgt_c) ** 2))
    return aligned, error


def plot_multi_encoder_tsne(
    embeddings_dict: Dict[str, np.ndarray],
    labels: Optional[List[str]] = None,
    method: str = "tsne",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot t-SNE/UMAP of multiple encoders side by side.

    Args:
        embeddings_dict: {encoder_name: embeddings (n, d)}
        labels: Category labels for each sample
        method: 'tsne' or 'umap'
        output_path: Save path
    """
    n_encoders = len(embeddings_dict)
    fig, axes = create_figure(1, n_encoders, figsize=(6 * n_encoders, 5))
    if n_encoders == 1:
        axes = [axes]

    reducer = reduce_umap if method == "umap" else reduce_tsne

    for ax, (name, emb) in zip(axes, embeddings_dict.items()):
        reduced = reducer(emb)
        color = get_model_color(name)

        if labels:
            unique = sorted(set(labels))
            for lbl in unique:
                mask = [l == lbl for l in labels]
                ax.scatter(
                    reduced[mask, 0], reduced[mask, 1],
                    c=get_data_type_color(lbl), label=lbl, s=40, alpha=0.7,
                )
            ax.legend(fontsize=9)
        else:
            ax.scatter(reduced[:, 0], reduced[:, 1], c=color, s=40, alpha=0.7)

        style_axis(ax, title=f"{name.upper()} ({method.upper()})", grid=False)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_procrustes_comparison(
    embeddings_dict: Dict[str, np.ndarray],
    target_name: str,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Show before/after Procrustes alignment to a target space.

    Args:
        embeddings_dict: {encoder_name: embeddings}
        target_name: Which encoder is the target
        output_path: Save path
    """
    target_emb = embeddings_dict[target_name]
    others = {k: v for k, v in embeddings_dict.items() if k != target_name}

    n = len(others)
    fig, axes = create_figure(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = [axes]

    for i, (name, src_emb) in enumerate(others.items()):
        aligned, error = procrustes_align(src_emb, target_emb)

        # Before
        src_2d = reduce_tsne(src_emb)
        tgt_2d = reduce_tsne(target_emb)
        axes[i][0].scatter(src_2d[:, 0], src_2d[:, 1], c=get_model_color(name), label=name, alpha=0.6, s=30)
        axes[i][0].scatter(tgt_2d[:, 0], tgt_2d[:, 1], c=get_model_color(target_name), label=target_name, alpha=0.6, s=30)
        style_axis(axes[i][0], title=f"Before Alignment ({name} vs {target_name})", grid=False)
        axes[i][0].legend()

        # After
        combined = np.vstack([aligned, target_emb[:, :aligned.shape[1]] - target_emb[:, :aligned.shape[1]].mean(axis=0)])
        combined_2d = reduce_tsne(combined)
        mid = len(aligned)
        axes[i][1].scatter(combined_2d[:mid, 0], combined_2d[:mid, 1], c=get_model_color(name), label=f"{name} (aligned)", alpha=0.6, s=30)
        axes[i][1].scatter(combined_2d[mid:, 0], combined_2d[mid:, 1], c=get_model_color(target_name), label=target_name, alpha=0.6, s=30)
        style_axis(axes[i][1], title=f"After Procrustes (error={error:.2f})", grid=False)
        axes[i][1].legend()

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig


def plot_similarity_matrices(
    embeddings_dict: Dict[str, np.ndarray],
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot cosine similarity matrices and Spearman correlations.

    Args:
        embeddings_dict: {encoder_name: normalized embeddings}
    """
    import seaborn as sns

    names = list(embeddings_dict.keys())
    n = len(names)

    fig, axes = create_figure(1, n + 1, figsize=(5 * (n + 1), 4))

    sim_matrices = {}
    for i, name in enumerate(names):
        emb = embeddings_dict[name]
        sim = emb @ emb.T
        sim_matrices[name] = sim
        sns.heatmap(sim, ax=axes[i], cmap="coolwarm", vmin=-1, vmax=1, square=True)
        style_axis(axes[i], title=f"{name.upper()} Similarity", grid=False)

    # Spearman correlation matrix
    corr_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            triu = np.triu_indices(sim_matrices[names[i]].shape[0], k=1)
            r, _ = spearmanr(sim_matrices[names[i]][triu], sim_matrices[names[j]][triu])
            corr_matrix[i, j] = corr_matrix[j, i] = r

    sns.heatmap(corr_matrix, ax=axes[-1], annot=True, fmt=".3f",
                xticklabels=[n.upper() for n in names],
                yticklabels=[n.upper() for n in names],
                cmap="YlOrRd", vmin=0, vmax=1, square=True)
    style_axis(axes[-1], title="Spearman Correlation", grid=False)

    plt.tight_layout()
    if output_path:
        save_figure(fig, output_path)
    return fig

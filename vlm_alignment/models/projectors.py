"""Unified projector architectures and training utilities.

Consolidates 6 duplicated projector implementations into a single module.
Provides: LinearProjection, MLPProjection, TwoLayerMLPProjection,
CrossAttentionProjection, and a unified train_projector() function.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


# ============================================================
# Projector Architectures
# ============================================================


class ProjectionLayer(nn.Module):
    """Base class for projection layers."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


class LinearProjection(ProjectionLayer):
    """Simple linear projection."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProjection(ProjectionLayer):
    """2-layer MLP with GELU (LLaVA-style)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__(input_dim, output_dim)
        if hidden_dim is None:
            hidden_dim = output_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TwoLayerMLPProjection(ProjectionLayer):
    """2-layer MLP with LayerNorm (controlled projector for fair comparison)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None):
        super().__init__(input_dim, output_dim)
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CrossAttentionProjection(ProjectionLayer):
    """Cross-attention based projection (Q-Former style)."""

    def __init__(
        self, input_dim: int, output_dim: int, num_queries: int = 32, num_heads: int = 8
    ):
        super().__init__(input_dim, output_dim)
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, output_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True,
        )
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size = x.shape[0]
        queries = self.queries.expand(batch_size, -1, -1)
        attn_out, _ = self.cross_attn(queries, x, x)
        attn_out = self.norm(attn_out)
        return attn_out.mean(dim=1)


# ============================================================
# Factory & Training
# ============================================================

PROJECTOR_TYPES = {
    "linear": LinearProjection,
    "mlp": MLPProjection,
    "2layer_mlp": TwoLayerMLPProjection,
    "cross_attention": CrossAttentionProjection,
}


def create_projector(
    projection_type: str, input_dim: int, output_dim: int, **kwargs
) -> ProjectionLayer:
    """Create a projector by type name.

    Args:
        projection_type: One of 'linear', 'mlp', '2layer_mlp', 'cross_attention'
        input_dim: Input embedding dimension
        output_dim: Output (target) embedding dimension
    """
    if projection_type not in PROJECTOR_TYPES:
        raise ValueError(
            f"Unknown projector: {projection_type}. Available: {list(PROJECTOR_TYPES.keys())}"
        )
    return PROJECTOR_TYPES[projection_type](input_dim, output_dim, **kwargs)


@dataclass
class ProjectorTrainResult:
    """Results from projector training."""

    projector: ProjectionLayer
    train_loss: float
    test_loss: Optional[float]
    train_cka: Optional[float]
    test_cka: Optional[float]
    train_cosine: Optional[float]
    test_cosine: Optional[float]


def train_projector(
    source_embeddings: np.ndarray,
    target_embeddings: np.ndarray,
    projection_type: str = "2layer_mlp",
    epochs: int = 300,
    lr: float = 1e-3,
    test_ratio: float = 0.2,
    device: str = "cuda",
    compute_metrics: bool = True,
) -> ProjectorTrainResult:
    """Train a projector to align source to target embeddings.

    This unified function replaces 6 duplicated training loops across the
    original codebase, supporting train/test split and multiple metrics.

    Args:
        source_embeddings: Source embeddings (n, d_source)
        target_embeddings: Target embeddings (n, d_target)
        projection_type: Projector type name
        epochs: Number of training epochs
        lr: Learning rate
        test_ratio: Fraction of data for testing (0 to disable split)
        device: torch device
        compute_metrics: Whether to compute CKA and cosine similarity

    Returns:
        ProjectorTrainResult with trained projector and metrics
    """
    n = source_embeddings.shape[0]
    input_dim = source_embeddings.shape[1]
    output_dim = target_embeddings.shape[1]

    # Train/test split
    if test_ratio > 0 and n > 10:
        split = int(n * (1 - test_ratio))
        indices = np.random.permutation(n)
        train_idx, test_idx = indices[:split], indices[split:]
        src_train = torch.FloatTensor(source_embeddings[train_idx]).to(device)
        tgt_train = torch.FloatTensor(target_embeddings[train_idx]).to(device)
        src_test = torch.FloatTensor(source_embeddings[test_idx]).to(device)
        tgt_test = torch.FloatTensor(target_embeddings[test_idx]).to(device)
    else:
        src_train = torch.FloatTensor(source_embeddings).to(device)
        tgt_train = torch.FloatTensor(target_embeddings).to(device)
        src_test = tgt_test = None

    projector = create_projector(projection_type, input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(projector.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    projector.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        projected = projector(src_train)
        loss = criterion(projected, tgt_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    projector.eval()
    with torch.no_grad():
        train_pred = projector(src_train)
        train_loss = criterion(train_pred, tgt_train).item()

        test_loss = None
        if src_test is not None:
            test_pred = projector(src_test)
            test_loss = criterion(test_pred, tgt_test).item()

    train_cka = test_cka = train_cosine = test_cosine = None

    if compute_metrics:
        from vlm_alignment.analysis.cka import CKA

        with torch.no_grad():
            train_proj_np = train_pred.cpu().numpy()
            tgt_train_np = tgt_train.cpu().numpy()
            train_cka = CKA.compute_cka(train_proj_np, tgt_train_np)

            # Cosine similarity
            t_norm = train_proj_np / (np.linalg.norm(train_proj_np, axis=1, keepdims=True) + 1e-8)
            g_norm = tgt_train_np / (np.linalg.norm(tgt_train_np, axis=1, keepdims=True) + 1e-8)
            train_cosine = float(np.mean(np.sum(t_norm * g_norm, axis=1)))

            if src_test is not None:
                test_proj_np = test_pred.cpu().numpy()
                tgt_test_np = tgt_test.cpu().numpy()
                test_cka = CKA.compute_cka(test_proj_np, tgt_test_np)

                t_norm = test_proj_np / (np.linalg.norm(test_proj_np, axis=1, keepdims=True) + 1e-8)
                g_norm = tgt_test_np / (np.linalg.norm(tgt_test_np, axis=1, keepdims=True) + 1e-8)
                test_cosine = float(np.mean(np.sum(t_norm * g_norm, axis=1)))

    return ProjectorTrainResult(
        projector=projector,
        train_loss=train_loss,
        test_loss=test_loss,
        train_cka=train_cka,
        test_cka=test_cka,
        train_cosine=train_cosine,
        test_cosine=test_cosine,
    )

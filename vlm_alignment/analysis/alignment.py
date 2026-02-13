"""Alignment analysis including ELAS (Encoder-LLM Alignment Score).

Combines AlignmentAnalyzer and ELAS scoring into a unified module.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from vlm_alignment.analysis.cka import CKA
from vlm_alignment.models.projectors import train_projector


@dataclass
class AlignmentMetrics:
    """Metrics for a single encoder-LLM alignment."""

    cka_linear: float
    cka_rbf: float
    projection_mse: float
    cosine_after_proj: float


@dataclass
class ELASScore:
    """ELAS (Encoder-LLM Alignment Score) result."""

    score: float
    cka: float
    mse_norm: float
    gen_gap: float
    cosine_sim: float
    encoder: str
    llm: str


class AlignmentAnalyzer:
    """Analyzer for Vision Encoder - LLM alignment research."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    def compute_alignment_metrics(
        self,
        vision_embeddings: Dict[str, np.ndarray],
        text_embeddings: np.ndarray,
    ) -> Dict[str, AlignmentMetrics]:
        """Compute alignment metrics between vision encoders and text embeddings.

        Args:
            vision_embeddings: Dict mapping encoder name to embeddings (n, d_v)
            text_embeddings: Text embeddings (n, d_t)

        Returns:
            Dict mapping encoder name to AlignmentMetrics
        """
        results = {}

        for name, v_emb in vision_embeddings.items():
            # CKA scores
            cka_lin = CKA.compute_cka(v_emb, text_embeddings, kernel="linear")
            cka_rbf = CKA.compute_cka(v_emb, text_embeddings, kernel="rbf")

            # Projection via Ridge regression
            ridge = Ridge(alpha=1.0)
            ridge.fit(v_emb, text_embeddings)
            projected = ridge.predict(v_emb)

            mse = float(mean_squared_error(text_embeddings, projected))

            # Cosine similarity after projection
            proj_norm = projected / (np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8)
            text_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
            cosine = float(np.mean(np.sum(proj_norm * text_norm, axis=1)))

            results[name] = AlignmentMetrics(
                cka_linear=cka_lin,
                cka_rbf=cka_rbf,
                projection_mse=mse,
                cosine_after_proj=cosine,
            )

        return results

    def evaluate_projectors(
        self,
        vision_embeddings: Dict[str, np.ndarray],
        text_embeddings: np.ndarray,
        projection_types: Optional[List[str]] = None,
        epochs: int = 300,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate different projection strategies for each encoder.

        Returns:
            Nested dict: {encoder: {projector_type: loss}}
        """
        if projection_types is None:
            projection_types = ["linear", "mlp", "2layer_mlp"]

        results = {}
        for enc_name, v_emb in vision_embeddings.items():
            results[enc_name] = {}
            for proj_type in projection_types:
                result = train_projector(
                    v_emb,
                    text_embeddings,
                    projection_type=proj_type,
                    epochs=epochs,
                    device=self.device,
                    test_ratio=0.2,
                )
                results[enc_name][proj_type] = result.train_loss
        return results

    def compute_elas(
        self,
        vision_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        encoder_name: str,
        llm_name: str,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.2,
        epochs: int = 300,
    ) -> ELASScore:
        """Compute ELAS (Encoder-LLM Alignment Score).

        ELAS = alpha*CKA + beta*(1-MSE_norm) + gamma*(1-GenGap) + delta*CosSim

        Args:
            vision_embeddings: Vision encoder embeddings (n, d_v)
            text_embeddings: LLM text embeddings (n, d_t)
            encoder_name: Encoder name for labeling
            llm_name: LLM name for labeling
            alpha/beta/gamma/delta: Weight coefficients (must sum to 1)

        Returns:
            ELASScore with composite score and components
        """
        result = train_projector(
            vision_embeddings,
            text_embeddings,
            projection_type="2layer_mlp",
            epochs=epochs,
            device=self.device,
            test_ratio=0.2,
            compute_metrics=True,
        )

        cka = result.train_cka or 0.0
        mse_norm = min(result.train_loss, 1.0)  # Clamp to [0, 1]
        gen_gap = abs((result.test_loss or 0) - result.train_loss)
        cosine = result.train_cosine or 0.0

        score = alpha * cka + beta * (1 - mse_norm) + gamma * (1 - gen_gap) + delta * cosine

        return ELASScore(
            score=score,
            cka=cka,
            mse_norm=mse_norm,
            gen_gap=gen_gap,
            cosine_sim=cosine,
            encoder=encoder_name,
            llm=llm_name,
        )

    def compute_elas_matrix(
        self,
        vision_embeddings_dict: Dict[str, np.ndarray],
        text_embeddings_dict: Dict[str, np.ndarray],
    ) -> List[ELASScore]:
        """Compute ELAS for all encoder x LLM combinations.

        Args:
            vision_embeddings_dict: {encoder_name: embeddings}
            text_embeddings_dict: {llm_name: embeddings}

        Returns:
            List of ELASScore for each combination
        """
        results = []
        for enc_name, v_emb in vision_embeddings_dict.items():
            for llm_name, t_emb in text_embeddings_dict.items():
                # Ensure same number of samples
                n = min(v_emb.shape[0], t_emb.shape[0])
                score = self.compute_elas(
                    v_emb[:n], t_emb[:n], enc_name, llm_name
                )
                results.append(score)
                print(f"  ELAS({enc_name}, {llm_name}) = {score.score:.4f}")
        return results

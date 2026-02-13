"""Deep CORAL: Correlation Alignment for Domain Adaptation.

Compares text-image, text-text, and image-image alignment by matching
second-order statistics (covariance matrices) between representation spaces.

Reference: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep Domain
Adaptation", ECCV 2016. https://arxiv.org/abs/1607.01719

Key difference from CKA:
- CKA measures structural similarity (kernel alignment) but ignores distribution
  shape. High CKA does NOT guarantee good task performance (CKA-Performance Paradox).
- CORAL directly aligns covariance matrices, capturing how features co-vary and
  their distributional spread. This gives a more actionable alignment signal.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


def compute_covariance(features: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute covariance matrix of features.

    Args:
        features: [N, D] feature matrix
        eps: Numerical stability constant

    Returns:
        [D, D] covariance matrix
    """
    N, D = features.shape
    mean = features.mean(axis=0, keepdims=True)
    centered = features - mean
    cov = (centered.T @ centered) / max(N - 1, 1)
    cov += eps * np.eye(D)
    return cov


def coral_distance(
    source: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Compute CORAL distance between two feature sets.

    L_CORAL = (1 / 4d^2) * ||C_s - C_t||^2_F

    Args:
        source: [N_s, D] source features
        target: [N_t, D] target features (must have same D)
        eps: Numerical stability

    Returns:
        CORAL distance (lower = better alignment)
    """
    d = source.shape[1]
    C_s = compute_covariance(source, eps=eps)
    C_t = compute_covariance(target, eps=eps)
    diff = C_s - C_t
    return float(np.sum(diff * diff) / (4 * d * d))


def coral_similarity(
    source: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """CORAL-based similarity score in [0, 1] (higher = better alignment).

    Transforms CORAL distance via: score = 1 / (1 + distance)
    """
    dist = coral_distance(source, target, eps=eps)
    return 1.0 / (1.0 + dist)


def align_dimensions(
    X: np.ndarray, Y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two matrices to the same dimensionality using PCA.

    Projects both to min(d_x, d_y, n-1) dimensions.

    Args:
        X: [N, D_x] first matrix
        Y: [N, D_y] second matrix

    Returns:
        (X_proj, Y_proj) both [N, common_dim]
    """
    if X.shape[1] == Y.shape[1]:
        return X, Y

    from sklearn.decomposition import PCA

    common_dim = min(X.shape[1], Y.shape[1], X.shape[0] - 1)
    pca_x = PCA(n_components=common_dim)
    pca_y = PCA(n_components=common_dim)
    return pca_x.fit_transform(X), pca_y.fit_transform(Y)


def eigenvalue_spectrum(cov: np.ndarray) -> np.ndarray:
    """Compute sorted eigenvalue spectrum of a covariance matrix.

    Args:
        cov: [D, D] covariance matrix

    Returns:
        Sorted eigenvalues (descending)
    """
    eigenvalues = np.linalg.eigvalsh(cov)
    return np.sort(eigenvalues)[::-1]


def spectral_divergence(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """Compute spectral divergence between two covariance matrices.

    Measures how different the eigenvalue distributions are.
    """
    eig_a = eigenvalue_spectrum(cov_a)
    eig_b = eigenvalue_spectrum(cov_b)

    # Truncate to same length
    min_len = min(len(eig_a), len(eig_b))
    eig_a = eig_a[:min_len]
    eig_b = eig_b[:min_len]

    # Normalize to probability distributions
    eig_a = np.abs(eig_a) + 1e-10
    eig_b = np.abs(eig_b) + 1e-10
    eig_a /= eig_a.sum()
    eig_b /= eig_b.sum()

    # Symmetric KL divergence
    kl_ab = np.sum(eig_a * np.log(eig_a / eig_b))
    kl_ba = np.sum(eig_b * np.log(eig_b / eig_a))
    return float((kl_ab + kl_ba) / 2)


@dataclass
class CORALMetrics:
    """CORAL analysis results for a single pair comparison."""

    coral_distance: float
    coral_similarity: float
    spectral_divergence: float
    mean_distance: float
    covariance_source: np.ndarray = field(repr=False)
    covariance_target: np.ndarray = field(repr=False)
    eigenvalues_source: np.ndarray = field(repr=False)
    eigenvalues_target: np.ndarray = field(repr=False)


@dataclass
class CrossModalResult:
    """Cross-modal comparison result (e.g., text vs image)."""

    encoder_name: str
    llm_name: str
    coral: CORALMetrics
    cka_linear: float
    cka_rbf: float


@dataclass
class IntraModalResult:
    """Intra-modal comparison result (e.g., encoder A vs encoder B)."""

    name_a: str
    name_b: str
    modality: str  # 'vision' or 'text'
    coral: CORALMetrics
    cka_linear: float


@dataclass
class EASResult:
    """Enhanced Alignment Score (EAS) combining CKA + CORAL + Discriminability."""

    encoder_name: str
    llm_name: str
    eas: float
    cka: float
    coral_score: float
    coral_raw: float
    discriminability: float


class CORALAnalyzer:
    """Deep CORAL-based alignment analyzer.

    Supports three comparison modes:
    1. Cross-modal: vision embeddings vs text embeddings
    2. Intra-modal vision: encoder A vs encoder B (same images)
    3. Intra-modal text: LLM A vs LLM B (same texts)

    Also computes Enhanced Alignment Score (EAS) that combines
    CKA, CORAL, and discriminability for a balanced metric.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def compute_coral_metrics(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> CORALMetrics:
        """Compute full CORAL metrics between two feature sets.

        Args:
            source: [N, D_s] source features
            target: [N, D_t] target features (dimensions aligned automatically)

        Returns:
            CORALMetrics with distance, similarity, spectra, covariances
        """
        src, tgt = align_dimensions(source, target)

        cov_s = compute_covariance(src, self.eps)
        cov_t = compute_covariance(tgt, self.eps)

        dist = coral_distance(src, tgt, self.eps)
        sim = 1.0 / (1.0 + dist)
        spec_div = spectral_divergence(cov_s, cov_t)

        # Mean distance (first-order statistic)
        mean_s = src.mean(axis=0)
        mean_t = tgt.mean(axis=0)
        mean_dist = float(np.linalg.norm(mean_s - mean_t))

        eig_s = eigenvalue_spectrum(cov_s)
        eig_t = eigenvalue_spectrum(cov_t)

        return CORALMetrics(
            coral_distance=dist,
            coral_similarity=sim,
            spectral_divergence=spec_div,
            mean_distance=mean_dist,
            covariance_source=cov_s,
            covariance_target=cov_t,
            eigenvalues_source=eig_s,
            eigenvalues_target=eig_t,
        )

    def cross_modal_comparison(
        self,
        vision_embeddings: Dict[str, np.ndarray],
        text_embeddings: np.ndarray,
        llm_name: str = "llm",
    ) -> Dict[str, CrossModalResult]:
        """Compare each vision encoder against text embeddings (cross-modal).

        This is the primary use case: measuring how well each encoder's
        visual representation aligns with the LLM's text representation.

        Args:
            vision_embeddings: {encoder_name: [N, D_v]}
            text_embeddings: [N, D_t]
            llm_name: Name of the LLM for labeling

        Returns:
            {encoder_name: CrossModalResult}
        """
        from vlm_alignment.analysis.cka import CKA

        results = {}
        for enc_name, v_emb in vision_embeddings.items():
            n = min(v_emb.shape[0], text_embeddings.shape[0])
            v = v_emb[:n]
            t = text_embeddings[:n]

            coral_metrics = self.compute_coral_metrics(v, t)
            cka_lin = CKA.compute_cka(v, t, kernel="linear")
            cka_rbf = CKA.compute_cka(v, t, kernel="rbf")

            results[enc_name] = CrossModalResult(
                encoder_name=enc_name,
                llm_name=llm_name,
                coral=coral_metrics,
                cka_linear=cka_lin,
                cka_rbf=cka_rbf,
            )

        return results

    def intra_modal_comparison(
        self,
        embeddings: Dict[str, np.ndarray],
        modality: str = "vision",
    ) -> List[IntraModalResult]:
        """Compare embeddings within the same modality (encoder vs encoder).

        Useful for understanding how similar different encoders' representations
        are to each other.

        Args:
            embeddings: {model_name: [N, D]}
            modality: 'vision' or 'text'

        Returns:
            List of IntraModalResult for each pair
        """
        from vlm_alignment.analysis.cka import CKA

        names = list(embeddings.keys())
        results = []

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = embeddings[names[i]], embeddings[names[j]]
                n = min(a.shape[0], b.shape[0])
                a, b = a[:n], b[:n]

                coral_metrics = self.compute_coral_metrics(a, b)
                cka_lin = CKA.compute_cka(a, b, kernel="linear")

                results.append(IntraModalResult(
                    name_a=names[i],
                    name_b=names[j],
                    modality=modality,
                    coral=coral_metrics,
                    cka_linear=cka_lin,
                ))

        return results

    def compute_eas(
        self,
        vision_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
        encoder_name: str,
        llm_name: str,
        labels: Optional[np.ndarray] = None,
        alpha: float = 0.3,
        beta: float = 0.3,
        gamma: float = 0.4,
    ) -> EASResult:
        """Compute Enhanced Alignment Score (EAS).

        EAS = alpha * CKA + beta * CORAL_sim + gamma * Discriminability

        Unlike CKA alone, EAS also considers:
        - CORAL: distribution alignment (second-order statistics)
        - Discriminability: whether features maintain class separability

        Args:
            vision_embeddings: [N, D_v]
            text_embeddings: [N, D_t]
            encoder_name: Encoder name
            llm_name: LLM name
            labels: Optional class labels for discriminability
            alpha/beta/gamma: Weight coefficients (should sum to 1)

        Returns:
            EASResult with composite score and components
        """
        from vlm_alignment.analysis.cka import CKA

        n = min(vision_embeddings.shape[0], text_embeddings.shape[0])
        v = vision_embeddings[:n]
        t = text_embeddings[:n]

        # CKA
        cka = CKA.compute_cka(v, t, kernel="linear")

        # CORAL
        v_aligned, t_aligned = align_dimensions(v, t)
        coral_raw = coral_distance(v_aligned, t_aligned, self.eps)
        coral_score = 1.0 / (1.0 + coral_raw)

        # Discriminability
        if labels is not None:
            labels = labels[:n]
            disc = self._compute_discriminability(v_aligned, labels)
        else:
            disc = self._estimate_discriminability(v_aligned)

        eas = alpha * cka + beta * coral_score + gamma * disc

        return EASResult(
            encoder_name=encoder_name,
            llm_name=llm_name,
            eas=eas,
            cka=cka,
            coral_score=coral_score,
            coral_raw=coral_raw,
            discriminability=disc,
        )

    def _compute_discriminability(
        self, features: np.ndarray, labels: np.ndarray
    ) -> float:
        """Fisher discriminant ratio from labeled data."""
        unique = np.unique(labels)
        if len(unique) < 2:
            return 0.5

        global_mean = features.mean(axis=0)
        within_scatter = 0.0
        between_scatter = 0.0

        for label in unique:
            mask = labels == label
            cls_features = features[mask]
            cls_mean = cls_features.mean(axis=0)
            n_cls = cls_features.shape[0]

            within_scatter += np.sum((cls_features - cls_mean) ** 2)
            between_scatter += n_cls * np.sum((cls_mean - global_mean) ** 2)

        fisher = between_scatter / (within_scatter + 1e-8)
        # Normalize to [0, 1]
        return float(fisher / (fisher + 1))

    def _estimate_discriminability(self, features: np.ndarray) -> float:
        """Estimate discriminability without labels using embedding variance."""
        from sklearn.metrics.pairwise import cosine_similarity

        variance = np.var(features, axis=0).mean()
        sim_matrix = cosine_similarity(features)
        mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
        sim_variance = sim_matrix[mask].var()

        return float(min(1.0, (variance * 10 + sim_variance * 100) / 2))

    def full_analysis(
        self,
        vision_embeddings: Dict[str, np.ndarray],
        text_embeddings: np.ndarray,
        llm_name: str = "llm",
        labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run complete CORAL analysis: cross-modal, intra-modal, and EAS.

        Args:
            vision_embeddings: {encoder_name: [N, D_v]}
            text_embeddings: [N, D_t]
            llm_name: LLM name
            labels: Optional data type labels

        Returns:
            Dict with 'cross_modal', 'intra_modal', 'eas' keys
        """
        # Cross-modal: each encoder vs text
        cross_modal = self.cross_modal_comparison(
            vision_embeddings, text_embeddings, llm_name
        )

        # Intra-modal: encoder vs encoder
        intra_modal = self.intra_modal_comparison(vision_embeddings, "vision")

        # EAS for each encoder
        eas_results = {}
        for enc_name, v_emb in vision_embeddings.items():
            eas = self.compute_eas(
                v_emb, text_embeddings, enc_name, llm_name, labels
            )
            eas_results[enc_name] = eas

        return {
            "cross_modal": cross_modal,
            "intra_modal": intra_modal,
            "eas": eas_results,
        }

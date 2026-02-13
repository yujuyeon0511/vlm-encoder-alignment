"""Centered Kernel Alignment (CKA) implementation.

Measures similarity between two representation spaces, useful for
comparing vision encoder outputs with LLM embeddings.
"""

import numpy as np


class CKA:
    """CKA with linear and RBF kernel support."""

    @staticmethod
    def centering_matrix(n: int) -> np.ndarray:
        """Create centering matrix H = I - 1/n * 11^T."""
        return np.eye(n) - np.ones((n, n)) / n

    @staticmethod
    def linear_kernel(X: np.ndarray) -> np.ndarray:
        """Compute linear kernel K = XX^T."""
        return X @ X.T

    @staticmethod
    def rbf_kernel(X: np.ndarray, sigma: float = None) -> np.ndarray:
        """Compute RBF kernel."""
        sq_dists = (
            np.sum(X**2, axis=1, keepdims=True)
            + np.sum(X**2, axis=1)
            - 2 * X @ X.T
        )
        if sigma is None:
            sigma = np.sqrt(np.median(sq_dists[sq_dists > 0]))
        return np.exp(-sq_dists / (2 * sigma**2))

    @staticmethod
    def hsic(K: np.ndarray, L: np.ndarray, H: np.ndarray) -> float:
        """Compute Hilbert-Schmidt Independence Criterion.

        HSIC(K, L) = 1/(n-1)^2 * tr(KHLH)
        """
        n = K.shape[0]
        return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)

    @classmethod
    def compute_cka(
        cls, X: np.ndarray, Y: np.ndarray, kernel: str = "linear"
    ) -> float:
        """Compute CKA between two representation matrices.

        Args:
            X: First representations (n_samples, n_features_1)
            Y: Second representations (n_samples, n_features_2)
            kernel: 'linear' or 'rbf'

        Returns:
            CKA similarity score in [0, 1], higher = more similar
        """
        assert X.shape[0] == Y.shape[0], "Number of samples must match"

        n = X.shape[0]
        H = cls.centering_matrix(n)

        if kernel == "linear":
            K = cls.linear_kernel(X)
            L = cls.linear_kernel(Y)
        elif kernel == "rbf":
            K = cls.rbf_kernel(X)
            L = cls.rbf_kernel(Y)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        hsic_kl = cls.hsic(K, L, H)
        hsic_kk = cls.hsic(K, K, H)
        hsic_ll = cls.hsic(L, L, H)

        if hsic_kk * hsic_ll == 0:
            return 0.0

        return hsic_kl / np.sqrt(hsic_kk * hsic_ll)

    @classmethod
    def compute_pairwise(
        cls, embeddings: dict, kernel: str = "linear"
    ) -> dict:
        """Compute pairwise CKA between multiple embedding sets.

        Args:
            embeddings: Dict mapping name to embedding array
            kernel: 'linear' or 'rbf'

        Returns:
            Dict mapping 'nameA vs nameB' to CKA score
        """
        results = {}
        names = list(embeddings.keys())
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if i < j:
                    score = cls.compute_cka(embeddings[n1], embeddings[n2], kernel)
                    results[f"{n1} vs {n2}"] = score
        return results

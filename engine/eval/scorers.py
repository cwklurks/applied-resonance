"""Anomaly scorers fit on NORMAL embeddings only.

Both scorers are deterministic and operate on dense float embeddings
(e.g. 2048-d PANNs vectors). Higher score = more anomalous. They are
designed for the high-dim / low-sample regime (d >> n).
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


def _validate_2d_finite(x: np.ndarray, name: str) -> np.ndarray:
    """Coerce to a 2-D float64 array and assert it is finite."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got {arr.ndim}-D shape {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values (no NaN/inf).")
    return arr


class KNNScorer:
    """Anomaly score = mean cosine distance to the k nearest NORMAL training
    embeddings. Fit ONLY on normal data; never pass abnormal clips to fit()."""

    def __init__(self, k: int = 2) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        self.k = k
        self._nn: NearestNeighbors | None = None

    def fit(self, x_normal: np.ndarray) -> "KNNScorer":
        arr = _validate_2d_finite(x_normal, "x_normal")
        n_samples = arr.shape[0]
        if n_samples < self.k:
            raise ValueError(
                f"KNNScorer needs n_samples >= k={self.k}, got {n_samples}."
            )
        self._nn = NearestNeighbors(
            n_neighbors=self.k, metric="cosine", algorithm="brute"
        ).fit(arr)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("KNNScorer.score() called before fit().")
        arr = _validate_2d_finite(x, "x")
        distances, _ = self._nn.kneighbors(arr, n_neighbors=self.k)
        return distances.mean(axis=1).astype(np.float64)


class MahalanobisScorer:
    """Anomaly score = Mahalanobis distance to the NORMAL training distribution,
    with Ledoit-Wolf shrinkage covariance (handles d >> n). Fit ONLY on normal
    data; never pass abnormal clips to fit()."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, x_normal: np.ndarray) -> "MahalanobisScorer":
        arr = _validate_2d_finite(x_normal, "x_normal")
        n_samples = arr.shape[0]
        if n_samples < 2:
            raise ValueError(
                f"MahalanobisScorer needs n_samples >= 2, got {n_samples}."
            )
        lw = LedoitWolf().fit(arr)
        self.mean_ = lw.location_.astype(np.float64)
        self.precision_ = lw.precision_.astype(np.float64)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisScorer.score() called before fit().")
        arr = _validate_2d_finite(x, "x")
        centered = arr - self.mean_
        squared = np.einsum("ij,jk,ik->i", centered, self.precision_, centered)
        squared = np.clip(squared, 0.0, None)
        return np.sqrt(squared).astype(np.float64)

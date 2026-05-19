# damd/core/reducers.py
"""Channel-dimensionality reducers for the forward-backward architecture.

Only two are needed for the two papers:

* :class:`PCA`         — data-adaptive projection via truncated SVD
* :class:`Precomputed` — user-supplied projection matrix

Both produce an orthonormal basis ``U ∈ R^{d×r}`` satisfying
``U^T U = I_r``, which is exactly the condition of Theorem 3 in the
MDAMD paper. Violating the orthonormality is allowed but lowers
detection accuracy (Proposition 4 / Corollary 6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .exceptions import ConfigError


# ----------------------------------------------------------------------
@dataclass
class ProjectionResult:
    U: np.ndarray              # (d, r) orthonormal basis
    singular_values: Optional[np.ndarray] = None     # (min(d, N),) or None
    energy_retention: Optional[float] = None         # η ∈ [0, 1]


# ----------------------------------------------------------------------
class PCA:
    """Truncated SVD-based projection.

    Given a signal ``X`` of shape ``(d, N)``, fit returns an orthonormal
    basis ``U ∈ R^{d×r}`` spanning the directions of greatest energy.
    The energy-retention fraction :math:`\\eta = \\sum_{i\\le r}\\sigma_i^2 /
    \\sum_i \\sigma_i^2` (Definition 7 of the MDAMD paper) is also
    reported.
    """

    def __init__(self, r: int, center: bool = False):
        if r < 1:
            raise ConfigError(f"r must be >= 1, got {r}")
        self.r = r
        self.center = center

    def fit(self, X: np.ndarray) -> ProjectionResult:
        X = np.atleast_2d(X).astype(np.float64)
        d, N = X.shape
        if self.r > d:
            raise ConfigError(
                f"r={self.r} exceeds number of channels d={d}"
            )
        if self.center:
            X = X - X.mean(axis=1, keepdims=True)

        # Economy SVD of X = U S V^T; columns of U are eigenvectors of XX^T
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        U_r = U[:, : self.r]
        total = float(np.sum(S ** 2))
        retained = float(np.sum(S[: self.r] ** 2))
        eta = retained / total if total > 0 else 1.0
        return ProjectionResult(
            U=U_r,
            singular_values=S,
            energy_retention=eta,
        )


class Precomputed:
    """Wrapper for a user-supplied projection matrix.

    Use this when the subspace is known a priori (e.g. a canonical
    correlation filter, or a fixed spatial pattern). The fit method
    performs orthogonality / normalisation checks and returns a
    :class:`ProjectionResult` whose ``energy_retention`` is left
    unspecified — the caller knows what their basis represents.
    """

    def __init__(self, U: np.ndarray, check: bool = True,
                 rtol: float = 1e-6):
        U = np.asarray(U, dtype=np.float64)
        if U.ndim != 2:
            raise ConfigError("U must be 2-D")
        self.U = U
        self.check = check
        self.rtol = rtol

    def fit(self, X: np.ndarray | None = None) -> ProjectionResult:
        d, r = self.U.shape
        if self.check:
            err = np.linalg.norm(self.U.T @ self.U - np.eye(r))
            if err > self.rtol * r:
                import warnings
                warnings.warn(
                    f"U^T U deviates from I by {err:.3g}; spectral "
                    "preservation may be violated (see Theorem 3).",
                    stacklevel=2,
                )
        return ProjectionResult(U=self.U)

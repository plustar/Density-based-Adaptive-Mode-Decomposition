# damd/core/bandwidth.py
"""Kernel-bandwidth estimation for meanshift clustering.

Three estimators are provided, corresponding to Section II-C-3 of
the DAMD paper:

- ``silverman``   — eq. (40), optimal under Gaussian assumption
- ``adaptive``    — eq. (43), curvature-based local refinement
- ``percentile``  — eq. (45), robust grid-adaptive estimator

All three accept non-uniform frequency grids (e.g. log-spaced CWT).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

EPS = 1e-12


# ----------------------------------------------------------------------
# Primitives
# ----------------------------------------------------------------------
def weighted_std(freqs: np.ndarray, weights: np.ndarray) -> float:
    """Weighted standard deviation of ``freqs`` with given ``weights``."""
    total = float(np.sum(weights))
    if total < EPS:
        return 0.0
    p = weights / total
    mu = float(np.sum(freqs * p))
    var = float(np.sum(p * (freqs - mu) ** 2))
    return float(np.sqrt(max(var, 0.0)))


def local_spacing(freqs: np.ndarray) -> Tuple[np.ndarray, float, bool]:
    """Per-bin local frequency spacing.

    Returns
    -------
    df_local : (N,) per-bin spacing (midpoint differences on interior,
               one-sided on edges)
    df_scalar : median spacing
    is_nonuniform : True if max/min > 2
    """
    n = len(freqs)
    if n <= 1:
        return np.array([0.01]), 0.01, False
    d = np.diff(freqs)
    df = np.empty(n)
    df[0] = d[0]
    df[-1] = d[-1]
    df[1:-1] = 0.5 * (d[:-1] + d[1:])
    df_scalar = float(np.median(df))
    nonu = (df.max() / max(df.min(), EPS)) > 2.0
    return df, df_scalar, nonu


# ----------------------------------------------------------------------
# The three public estimators
# ----------------------------------------------------------------------
def silverman(freqs: np.ndarray, weights: np.ndarray) -> float:
    """Silverman's rule, eq. (40) of the DAMD paper.

    ``h = (4/3)^(1/5) * σ_ω * n^(-1/5)`` ≈ ``0.9 σ n^(-0.2)``.
    """
    n = len(freqs)
    if n < 2:
        return 1e-3
    sigma = weighted_std(freqs, weights)
    bw = 0.9 * sigma * n ** (-0.2)
    return max(bw, 1e-6)


def percentile(freqs: np.ndarray, weights: np.ndarray,
               q: float = 10.0) -> float:
    """Percentile-based estimator, eq. (45).

    Takes the ``q``-th percentile of inter-frequency spacings, so that the
    bandwidth tracks the *finest* resolution of the grid — crucial for
    log-spaced CWT bases.
    """
    if len(freqs) < 2:
        return 1e-3
    d = np.abs(np.diff(np.sort(freqs)))
    d = d[d > EPS]
    if d.size == 0:
        return 1e-3
    return float(np.percentile(d, q))


def _curvature_lambda(freqs: np.ndarray, weights: np.ndarray,
                      df: float) -> np.ndarray:
    """λ_i = sqrt(1 / |∇²ρ|), normalised so min λ = 1.

    Per eq. (43)–(44) of the DAMD paper.
    """
    n = len(freqs)
    w = weights + EPS
    df2 = df * df
    d2 = np.zeros(n)
    d2[1:-1] = (w[2:] - 2 * w[1:-1] + w[:-2]) / df2
    d2[0] = d2[1]
    d2[-1] = d2[-2]
    curv = np.maximum(np.abs(d2), EPS)
    raw = np.sqrt(1.0 / curv)
    return raw / raw.min()


def adaptive(freqs: np.ndarray, weights: np.ndarray,
             h0: float | None = None) -> np.ndarray:
    """Adaptive per-bin bandwidth, eq. (43).

    Returns a per-bin vector rather than a scalar so that densely
    populated bands stay narrow while flat regions use larger kernels.
    """
    _, df, _ = local_spacing(freqs)
    if h0 is None:
        h0 = silverman(freqs, weights)
    lam = _curvature_lambda(freqs, weights, df)
    h_max = max(silverman(freqs, weights), h0)
    return np.clip(h0 * lam, h0, h_max)


# ----------------------------------------------------------------------
# Dispatcher
# ----------------------------------------------------------------------
def estimate_bandwidth(
    freqs: np.ndarray,
    weights: np.ndarray,
    rule: str = "percentile",
    scale: float = 1.0,
) -> Tuple[np.ndarray | float, float]:
    """Compute the kernel bandwidth used by meanshift.

    Parameters
    ----------
    freqs : (N,) frequency grid (normalised or in Hz, must be consistent
        with whatever meanshift is run on)
    weights : (N,) non-negative spectral weights
    rule : ``'silverman'`` | ``'adaptive'`` | ``'percentile'``
    scale : global multiplier applied after the rule

    Returns
    -------
    bw : float or (N,) per-bin array — used inside the meanshift loop
    bw_scalar : float — a representative scalar used for dedup / KDE
    """
    if rule == "silverman":
        h = silverman(freqs, weights) * scale
        return h, h
    if rule == "percentile":
        h = percentile(freqs, weights) * scale
        return h, h
    if rule == "adaptive":
        per_bin = adaptive(freqs, weights) * scale
        return per_bin, float(np.mean(per_bin))
    raise ValueError(f"unknown bandwidth rule '{rule}'")

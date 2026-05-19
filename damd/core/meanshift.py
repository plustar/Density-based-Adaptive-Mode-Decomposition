# damd/core/meanshift.py
"""Weighted mean-shift clustering on a 1-D power spectrum.

Implements Section II-C-1 of the DAMD paper:

* weighted kernel density estimation (eq. 25)
* mean-shift vector (eq. 27) iterated per seed (eq. 28)
* unique modes extracted by gap-split-mean deduplication (eq. 29)
* basin-of-attraction partition (eq. 30)

Nothing in this module is backend-specific: pure NumPy, scalar or
per-bin bandwidth both supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .bandwidth import EPS, estimate_bandwidth


# ----------------------------------------------------------------------
@dataclass
class ClusterResult:
    """One frame of meanshift output."""

    centers: np.ndarray       # (K,) cluster centres, sorted ascending
    bands: np.ndarray         # (K, 2) [start, end] indices in the freq grid
    labels: np.ndarray        # (N,) cluster id for every input bin, -1 if unassigned
    bandwidth: float          # scalar bandwidth used (mean of per-bin)


# ----------------------------------------------------------------------
def _meanshift_iteration(
    seeds: np.ndarray,
    freqs: np.ndarray,
    weights: np.ndarray,
    bw: np.ndarray | float,
    max_iter: int,
    tol: float,
    seed_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Run meanshift from ``seeds`` to convergence.

    Vectorised across all active seeds; each seed iterates independently.
    """
    centers = seeds.astype(np.float64).copy()
    active = np.ones(len(centers), dtype=bool)

    # Materialise per-seed bandwidth
    if np.ndim(bw) == 0:
        seed_bw = float(bw)
        use_array = False
    else:
        assert seed_idx is not None, "per-bin bw needs seed_idx"
        seed_bw = np.asarray(bw)[seed_idx][:, None]        # (S, 1)
        use_array = True

    for _ in range(max_iter):
        if not active.any():
            break
        idx = np.where(active)[0]
        # (S, N) distances
        d = centers[idx, None] - freqs[None, :]
        if use_array:
            h = seed_bw[idx]                                # (S, 1)
            k = np.exp(-0.5 * (d / h) ** 2) * weights[None, :]
        else:
            k = np.exp(-0.5 * (d / seed_bw) ** 2) * weights[None, :]
        tot = k.sum(axis=1)
        ok = tot > EPS
        new_c = centers[idx].copy()
        new_c[ok] = (k[ok] * freqs[None, :]).sum(axis=1) / tot[ok]

        shift = np.abs(new_c - centers[idx])
        active[idx[shift < tol]] = False
        active[idx[~ok]] = False
        centers[idx] = new_c
    return centers


def _dedup_scalar(pos: np.ndarray, h: float) -> np.ndarray:
    """Merge converged seeds whose gap is below ``h``."""
    if len(pos) == 0:
        return pos
    s = np.sort(pos)
    breaks = np.where(np.diff(s) > h)[0] + 1
    if len(breaks) == 0:
        return np.array([s.mean()])
    return np.array([g.mean() for g in np.split(s, breaks)])


def _dedup_adaptive(pos: np.ndarray, per_bin_bw: np.ndarray,
                    freqs: np.ndarray) -> np.ndarray:
    """Position-dependent merge threshold interpolated from per-bin bw."""
    if len(pos) == 0:
        return pos
    s = np.sort(pos)
    if len(s) == 1:
        return s
    mid = 0.5 * (s[:-1] + s[1:])
    local_h = np.interp(mid, freqs, per_bin_bw)
    breaks = np.where(np.diff(s) > local_h)[0] + 1
    if len(breaks) == 0:
        return np.array([s.mean()])
    return np.array([g.mean() for g in np.split(s, breaks)])


# ----------------------------------------------------------------------
def _assign_labels(freqs: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Voronoi-style basin assignment (eq. 30 of the DAMD paper).

    Under the hard-partition construction each frequency bin goes to the
    nearest centre — this is the argmin in eq. (20) of the MDAMD paper.
    """
    if len(centers) == 0:
        return -np.ones(len(freqs), dtype=np.int32)
    d = np.abs(freqs[:, None] - centers[None, :])   # (N, K)
    return np.argmin(d, axis=1).astype(np.int32)


def _labels_to_bands(labels: np.ndarray, K: int) -> np.ndarray:
    """Convert a label vector to (K, 2) [start, end] inclusive index bands.

    Because Voronoi cells on a sorted axis are contiguous, each cluster
    occupies a single interval. Empty clusters get [0, -1].
    """
    bands = np.full((K, 2), fill_value=-1, dtype=np.int64)
    for k in range(K):
        idx = np.where(labels == k)[0]
        if idx.size:
            bands[k, 0] = idx[0]
            bands[k, 1] = idx[-1]
    return bands


# ----------------------------------------------------------------------
def meanshift(
    power: np.ndarray,
    freqs: np.ndarray,
    *,
    bandwidth_rule: str = "percentile",
    bandwidth_scale: float = 1.0,
    seed_stride: int = 1,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> ClusterResult:
    """Cluster a 1-D power spectrum into modes via weighted meanshift.

    Parameters
    ----------
    power : (N,) non-negative spectral weights, e.g. :math:`|X(\\omega)|^2`
    freqs : (N,) frequency grid, strictly increasing (Hz or normalised)
    bandwidth_rule : ``'silverman'`` | ``'adaptive'`` | ``'percentile'``
    bandwidth_scale : global multiplier
    seed_stride : 1 → seed every bin (paper default), >1 → sub-sample
    max_iter, tol : meanshift inner-loop stopping criteria

    Returns
    -------
    :class:`ClusterResult`
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    N = len(freqs)
    if power.shape != (N,):
        raise ValueError("power and freqs must have the same length")

    # Normalise weights: meanshift is invariant to the global scale, but
    # we rescale to unit max for numerical stability in low-energy frames.
    w_max = power.max()
    weights = power / w_max if w_max > EPS else np.zeros_like(power)

    # Bandwidth
    bw, bw_scalar = estimate_bandwidth(freqs, weights, bandwidth_rule,
                                       bandwidth_scale)

    # Seeds
    stride = max(1, int(seed_stride))
    seed_idx = np.arange(0, N, stride)
    seeds = freqs[seed_idx]

    converged = _meanshift_iteration(
        seeds, freqs, weights, bw, max_iter, tol,
        seed_idx=seed_idx if np.ndim(bw) else None,
    )

    if np.ndim(bw):
        centers = _dedup_adaptive(converged, bw, freqs)
    else:
        centers = _dedup_scalar(converged, bw_scalar)

    centers = np.sort(centers)
    labels = _assign_labels(freqs, centers)
    bands = _labels_to_bands(labels, len(centers))

    return ClusterResult(
        centers=centers,
        bands=bands,
        labels=labels,
        bandwidth=float(bw_scalar),
    )


# ----------------------------------------------------------------------
def meanshift_2d(
    power: np.ndarray,
    freqs: np.ndarray,
    **kwargs,
) -> list[ClusterResult]:
    """Apply :func:`meanshift` independently to each column of ``power``.

    Parameters
    ----------
    power : (F, T) spectrogram-style power
    freqs : (F,) frequency grid

    Returns
    -------
    list of :class:`ClusterResult`, length ``T``.
    """
    if power.ndim != 2:
        raise ValueError("power must be (F, T)")
    return [meanshift(power[:, t], freqs, **kwargs)
            for t in range(power.shape[1])]

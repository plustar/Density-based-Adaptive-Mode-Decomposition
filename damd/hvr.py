# damd/hvr.py
"""Hybrid Variational Refinement (Section II-E-4 of the DAMD paper).

HVR uses DME results as initial centre frequencies for VMD, applying
variational refinement only when the quality gate of eq. (57) fails:

    :math:`\\Delta\\omega_{\\text{sep}}(k,t) < \\eta \\cdot h(t)`

where ``η`` is a scalar tolerance multiplier (``eta_refine``) applied
to the clustering bandwidth. This keeps HVR efficient — most frames
skip VMD entirely and retain the DME output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from .baselines.vmd import vmd as _vmd


@dataclass
class HVRResult:
    refined_centers: List[np.ndarray]    # length T, each (K_t,) in Hz
    refined_modes: List[np.ndarray]      # length T, each (K_t, Lw) real or empty
    refined_flags: np.ndarray            # (T,) bool: True if VMD was actually run
    iterations: np.ndarray               # (T,) per-frame VMD iterations (0 if skipped)


# ----------------------------------------------------------------------
def _min_gap(centers: np.ndarray) -> float:
    """Smallest pairwise gap — returns ``inf`` for a single centre."""
    if len(centers) < 2:
        return float("inf")
    s = np.sort(centers)
    return float(np.min(np.diff(s)))


def hvr(
    signal_frames: Sequence[np.ndarray],
    dme_centers: Sequence[np.ndarray],
    dme_bandwidth: Sequence[float],
    *,
    fs: float = 1.0,
    eta_refine: float = 1.0,
    alpha: float = 2000.0,
    tau: float = 0.0,
    max_iter: int = 50,
    tol: float = 1e-7,
) -> HVRResult:
    """Selectively refine DME modes with VMD.

    Parameters
    ----------
    signal_frames : sequence of length T, each item a 1-D signal of the
        current analysis window
    dme_centers : sequence of length T, each item the DME-detected
        centre frequencies in Hz
    dme_bandwidth : sequence of length T, scalar clustering bandwidths
    fs : sampling rate
    eta_refine : refinement trigger — run VMD when the closest-mode gap
        is below ``eta_refine * bandwidth``
    alpha, tau, max_iter, tol : passed through to VMD

    Returns
    -------
    :class:`HVRResult`
    """
    T = len(signal_frames)
    out_centers: List[np.ndarray] = []
    out_modes: List[np.ndarray] = []
    flags = np.zeros(T, dtype=bool)
    iters = np.zeros(T, dtype=np.int64)

    for t in range(T):
        x = np.asarray(signal_frames[t], dtype=np.float64).ravel()
        centers = np.asarray(dme_centers[t], dtype=np.float64).ravel()
        bw = float(dme_bandwidth[t])
        K = centers.size

        if K == 0:
            out_centers.append(centers)
            out_modes.append(np.zeros((0, x.size)))
            continue

        gap = _min_gap(centers)
        # Quality gate of eq. (57): refine only if any pair is too close
        if gap >= eta_refine * bw or K == 1:
            out_centers.append(centers)
            out_modes.append(np.zeros((K, x.size)))       # empty placeholder
            continue

        # Run VMD with DME initialisation
        try:
            res = _vmd(
                x, K,
                fs=fs, alpha=alpha, tau=tau,
                tol=tol, max_iter=max_iter,
                omega_init=centers / fs if fs else centers,
            )
        except Exception:
            out_centers.append(centers)
            out_modes.append(np.zeros((K, x.size)))
            continue

        out_centers.append(res.centers)
        out_modes.append(res.modes)
        flags[t] = True
        iters[t] = res.n_iter

    return HVRResult(
        refined_centers=out_centers,
        refined_modes=out_modes,
        refined_flags=flags,
        iterations=iters,
    )

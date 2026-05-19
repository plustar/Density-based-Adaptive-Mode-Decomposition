# damd/core/extract.py
"""Hard-partition mode extraction from frequency bands.

Given the Voronoi partition :math:`\\{B_k(t)\\}` produced by meanshift,
each mode is defined as

    :math:`\\hat u_k(\\omega,t) = X(\\omega,t) \\cdot \\mathbf{1}_{B_k(t)}(\\omega)`

(eq. 21 of the DAMD paper). This module implements that masking plus
two small conveniences:

* per-mode energy  (mean squared magnitude)
* optional inverse transform to get time-domain modes
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np


def apply_band_mask(
    tf: np.ndarray,
    bands: np.ndarray,
) -> np.ndarray:
    """Hard-mask a single frame.

    Parameters
    ----------
    tf : (d, F) or (F,) complex — one time frame
    bands : (K, 2) integer band boundaries [start, end] inclusive

    Returns
    -------
    modes : (K, d, F) or (K, F) complex — non-zero only on each band
    """
    single = tf.ndim == 1
    if single:
        tf = tf[None, :]
    d, F = tf.shape
    K = bands.shape[0]

    out = np.zeros((K, d, F), dtype=complex)
    for k in range(K):
        s, e = int(bands[k, 0]), int(bands[k, 1])
        if s < 0 or e < s:
            continue
        out[k, :, s : e + 1] = tf[:, s : e + 1]

    if single:
        out = out[:, 0, :]
    return out


def extract_modes_all_frames(
    tf: np.ndarray,
    per_frame_bands: Sequence[np.ndarray],
    *,
    keep_complex: bool = True,
) -> list[np.ndarray]:
    """Apply band masks frame-by-frame.

    Parameters
    ----------
    tf : (d, F, T) complex time-frequency representation
    per_frame_bands : length-T sequence, each item (K_t, 2)
    keep_complex : if False, keep only the real part of the mask output

    Returns
    -------
    list of length T, each item shape (K_t, d, F)
    """
    d, F, T = tf.shape
    result = []
    for t in range(T):
        bands = per_frame_bands[t]
        masked = apply_band_mask(tf[:, :, t], bands)
        if not keep_complex:
            masked = masked.real
        result.append(masked)
    return result


def mode_energy(
    tf: np.ndarray,
    per_frame_bands: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Per-mode, per-frame energy :math:`\\sum_{\\omega \\in B_k} |X|^2`.

    Parameters
    ----------
    tf : (d, F, T)
    per_frame_bands : length-T sequence of (K_t, 2)

    Returns
    -------
    list of length T, each item shape (K_t,) aggregated across channels.
    """
    P = np.sum(np.abs(tf) ** 2, axis=0)              # (F, T)
    out = []
    for t, bands in enumerate(per_frame_bands):
        col = P[:, t]
        K = bands.shape[0]
        e = np.zeros(K)
        for k in range(K):
            s, ee = int(bands[k, 0]), int(bands[k, 1])
            if s < 0 or ee < s:
                continue
            e[k] = col[s : ee + 1].sum()
        out.append(e)
    return out


def mode_centers_from_power(
    power: np.ndarray,
    freqs: np.ndarray,
    per_frame_bands: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Power-weighted centroid of every band (eq. 24 of the DAMD paper).

    This is used to *refine* the meanshift mode positions once the
    partition is fixed; it is also the closed-form K-means centroid
    update, and coincides with the MVMD ω-update of eq. (50) in the
    DAMD paper / eq. (12) of the MDAMD paper.
    """
    out = []
    for t, bands in enumerate(per_frame_bands):
        col = power[:, t]
        K = bands.shape[0]
        c = np.zeros(K)
        for k in range(K):
            s, e = int(bands[k, 0]), int(bands[k, 1])
            if s < 0 or e < s:
                continue
            w = col[s : e + 1]
            f = freqs[s : e + 1]
            z = w.sum()
            c[k] = (w * f).sum() / z if z > 0 else 0.5 * (f[0] + f[-1])
        out.append(c)
    return out


def filter_by_energy_percentile(
    per_frame_bands: Sequence[np.ndarray],
    per_frame_centers: Sequence[np.ndarray],
    per_frame_energy: Sequence[np.ndarray],
    percentile: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Drop modes whose energy is below the given percentile.

    The paper (DAMD §III-E) visualises results with an energy-stratified
    colour map and typically displays only the top 70% — call this with
    ``percentile=30`` to emulate that filter.
    """
    all_e = np.concatenate([e for e in per_frame_energy if e.size > 0])
    if all_e.size == 0:
        return list(per_frame_bands), list(per_frame_centers), list(per_frame_energy)
    threshold = float(np.percentile(all_e, percentile))

    kept_b, kept_c, kept_e = [], [], []
    for b, c, e in zip(per_frame_bands, per_frame_centers, per_frame_energy):
        if e.size == 0:
            kept_b.append(b); kept_c.append(c); kept_e.append(e); continue
        keep = e >= threshold
        kept_b.append(b[keep])
        kept_c.append(c[keep])
        kept_e.append(e[keep])
    return kept_b, kept_c, kept_e

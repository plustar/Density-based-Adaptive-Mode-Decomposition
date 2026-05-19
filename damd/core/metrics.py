# damd/core/metrics.py
"""Evaluation metrics used by the paper experiments.

* :func:`consistency`                — Definition 4 of the MDAMD paper
* :func:`match_centers`              — one-sided nearest-neighbour matching
* :func:`detection_metrics`          — precision / recall / F1 / MAE
* :func:`reconstruction_error`       — RMSE between signal and mode sum
"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Cross-channel consistency — MDAMD eq. (14)
# ----------------------------------------------------------------------
def consistency(tf: np.ndarray, centers: np.ndarray,
                freqs: np.ndarray) -> np.ndarray:
    r"""Cross-channel consistency at each detected centre frequency.

    .. math::
        \mathcal{C}_k(t) = \frac{|\sum_c X_c(\omega_k, t)|^2}
        {C \cdot \sum_c |X_c(\omega_k, t)|^2}

    Parameters
    ----------
    tf : (C, F, T) complex time-frequency map
    centers : (K,) centre frequencies in the same units as ``freqs``
    freqs : (F,) frequency grid

    Returns
    -------
    C_k : (K,) consistency scores in [0, 1]
    """
    if tf.ndim != 3:
        raise ValueError("tf must be (C, F, T); use np.atleast_3d if needed")
    C, F, T = tf.shape

    # Nearest-bin indices for each requested centre
    idx = np.clip(np.searchsorted(freqs, centers), 0, F - 1)
    X = tf[:, idx, :]                                       # (C, K, T)

    num = np.abs(X.sum(axis=0)) ** 2                        # (K, T)
    den = C * np.sum(np.abs(X) ** 2, axis=0)                # (K, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(den > 0, num / den, 0.0)
    # Mean over time — single scalar per mode
    return ratio.mean(axis=1)


# ----------------------------------------------------------------------
# Center-frequency matching
# ----------------------------------------------------------------------
def match_centers(
    detected: np.ndarray,
    truth: np.ndarray,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one match of detected centres to ground-truth.

    Returns
    -------
    matched_det : indices into ``detected`` that matched
    matched_tru : indices into ``truth`` that matched (same length)
    """
    detected = np.asarray(detected, dtype=float).ravel()
    truth = np.asarray(truth, dtype=float).ravel()
    if detected.size == 0 or truth.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    d = np.abs(detected[:, None] - truth[None, :])          # (D, T)
    used_t = np.zeros(truth.size, dtype=bool)
    used_d = np.zeros(detected.size, dtype=bool)
    matched_d, matched_t = [], []

    # Greedy: pick closest pair repeatedly
    flat = np.argsort(d.ravel())
    for idx in flat:
        i, j = np.unravel_index(idx, d.shape)
        if d[i, j] > tolerance:
            break
        if used_d[i] or used_t[j]:
            continue
        matched_d.append(i)
        matched_t.append(j)
        used_d[i] = True
        used_t[j] = True
    return np.array(matched_d, dtype=int), np.array(matched_t, dtype=int)


def detection_metrics(
    detected_per_frame: Sequence[np.ndarray],
    truth_per_frame: Sequence[np.ndarray],
    tolerance: float,
) -> dict:
    """Aggregate precision / recall / F1 / MAE across frames.

    Parameters
    ----------
    detected_per_frame, truth_per_frame : lists of length T, each entry a
        1-D array of centre frequencies (variable length)
    tolerance : matching radius in the same units as the centres

    Returns
    -------
    dict with keys ``'precision'``, ``'recall'``, ``'f1'``, ``'mae'``
    """
    tp = fp = fn = 0
    errors: list[float] = []
    for det, tru in zip(detected_per_frame, truth_per_frame):
        det = np.asarray(det).ravel()
        tru = np.asarray(tru).ravel()
        if det.size == 0 and tru.size == 0:
            continue
        if det.size == 0:
            fn += tru.size
            continue
        if tru.size == 0:
            fp += det.size
            continue
        md, mt = match_centers(det, tru, tolerance)
        tp += len(md)
        fp += det.size - len(md)
        fn += tru.size - len(mt)
        if len(md):
            errors.extend(np.abs(det[md] - tru[mt]).tolist())

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    mae = float(np.mean(errors)) if errors else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "mae": mae,
            "tp": tp, "fp": fp, "fn": fn}


# ----------------------------------------------------------------------
def reconstruction_rmse(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Root-mean-square error between a signal and its reconstruction."""
    d = np.asarray(x, dtype=float).ravel() - np.asarray(x_hat, dtype=float).ravel()
    return float(np.sqrt(np.mean(d ** 2)))

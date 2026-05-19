# damd/core/aggregate.py
"""Spectral aggregation :math:`P(\\omega) = s(\\omega)^H A\\, s(\\omega)`.

Implements Definition 2 of the MDAMD paper. Three concrete cases are
used throughout the paper experiments:

* ``A = 1``            — single-channel DAMD (scalar spectrum)
* ``A = I_d``          — MDAMD, sum of per-channel power
* ``A = U U^T``        — projected MDAMD (high-dim case in §V)

Because the final two cases are diagonal after projection, we do not
need a full matrix multiplication: we project once and then use the
identity form.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def aggregate_power(tf: np.ndarray) -> np.ndarray:
    """Aggregated power spectrum with ``A = I_d``.

    Parameters
    ----------
    tf : (d, F, T) complex time-frequency map — one slice per channel,
        or (F, T) for a single-channel signal.

    Returns
    -------
    P : (F, T) real non-negative spectrum.
    """
    tf = np.asarray(tf)
    if tf.ndim == 2:
        return np.abs(tf) ** 2
    if tf.ndim == 3:
        return np.sum(np.abs(tf) ** 2, axis=0)
    raise ValueError(f"tf must have rank 2 or 3, got {tf.ndim}")


def aggregate_projected(
    tf_original: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """Aggregated power under :math:`A = U U^T`.

    For *unitary* :math:`U` (the only case supported by Theorem 3), this
    equals :math:`\\|U^T s\\|_2^2`. We therefore project once in the
    channel dimension and then fall back to the identity aggregate.

    Parameters
    ----------
    tf_original : (d, F, T) complex time-frequency map of the original signal
    U : (d, r) projection matrix with ``U^T U = I_r``

    Returns
    -------
    P : (F, T)
    """
    if tf_original.ndim != 3:
        raise ValueError("tf_original must be (d, F, T)")
    if U.ndim != 2:
        raise ValueError("U must be (d, r)")

    d, F, T = tf_original.shape
    if U.shape[0] != d:
        raise ValueError(f"U has {U.shape[0]} rows, expected d={d}")

    # (d, F, T) -> (r, F, T)
    Y = np.einsum("dr,dft->rft", U, tf_original)
    return np.sum(np.abs(Y) ** 2, axis=0)


def project_signal(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Apply ``Y = U^T X`` in the time domain.

    Used by the forward phase of the forward-backward decomposition
    (Definition 3 of the MDAMD paper) when the transform is linear, so
    that :math:`T(U^T X) = U^T T(X)` and we can project before
    transforming.
    """
    X = np.atleast_2d(X)
    return U.T @ X

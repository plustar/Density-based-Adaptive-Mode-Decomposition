# damd/baselines/mvmd.py
"""Multivariate Variational Mode Decomposition (Rehman & Aftab, 2019)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._admm import admm_vmd


@dataclass
class MVMDResult:
    modes: np.ndarray          # (C, K, N) real time-domain modes per channel
    centers: np.ndarray        # (K,) shared centre frequencies in Hz
    n_iter: int
    convergence: float


def mvmd(
    X: np.ndarray,
    K: int,
    *,
    fs: float = 1.0,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    omega_init: Optional[np.ndarray] = None,
) -> MVMDResult:
    """Decompose a ``C``-channel signal into ``K`` modes with shared centres.

    Uses real-FFT per channel, then feeds the ``(C, F, 1)`` spectrum to
    the shared ADMM solver in non-dynamic mode. The centre-frequency
    update in the solver already aggregates power across channels,
    matching eq. (50) of the DAMD paper / eq. (12) of the MDAMD paper.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    C, N = X.shape
    Fhat = np.fft.rfft(X, axis=-1)                   # (C, F)
    freqs = np.fft.rfftfreq(N)
    f_hat = Fhat[:, :, None]                         # (C, F, 1)

    if omega_init is not None:
        omega_init = np.asarray(omega_init) / fs

    u, w, n_iter, conv = admm_vmd(
        f_hat, freqs, K,
        alpha=alpha, tau=tau, tol=tol,
        max_iter=max_iter, omega_init=omega_init,
        dynamic=False,
    )

    # Per-channel, per-mode irfft
    u_cf = u[:, :, :, 0]                             # (C, F, K)
    modes = np.fft.irfft(np.transpose(u_cf, (0, 2, 1)),
                         n=N, axis=-1)               # (C, K, N)

    return MVMDResult(
        modes=modes,
        centers=w * fs,
        n_iter=n_iter,
        convergence=conv,
    )

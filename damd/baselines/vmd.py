# damd/baselines/vmd.py
"""Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014).

A thin wrapper over the shared ADMM core. We operate on the one-sided
real-FFT spectrum: this avoids the conjugate-symmetry bookkeeping of
the original mirror-extension formulation while giving identical
numerical results for real signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ._admm import admm_vmd


@dataclass
class VMDResult:
    modes: np.ndarray          # (K, N) real time-domain modes
    centers: np.ndarray        # (K,) centre frequencies in Hz
    n_iter: int
    convergence: float


def vmd(
    x: np.ndarray,
    K: int,
    *,
    fs: float = 1.0,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    omega_init: Optional[np.ndarray] = None,
) -> VMDResult:
    """Decompose a 1-D real signal into ``K`` variational modes."""
    x = np.asarray(x, dtype=np.float64).ravel()
    N = x.size
    fhat = np.fft.rfft(x)                           # (F,)
    freqs = np.fft.rfftfreq(N)                      # normalised [0, 0.5]
    f_hat = fhat[None, :, None]                     # (1, F, 1)

    if omega_init is not None:
        omega_init = np.asarray(omega_init) / fs

    u, w, n_iter, conv = admm_vmd(
        f_hat, freqs, K,
        alpha=alpha, tau=tau, tol=tol,
        max_iter=max_iter, omega_init=omega_init,
        dynamic=False,
    )

    modes = np.fft.irfft(u[0, :, :, 0].T, n=N, axis=-1)     # (K, N)

    return VMDResult(
        modes=modes,
        centers=w * fs,
        n_iter=n_iter,
        convergence=conv,
    )

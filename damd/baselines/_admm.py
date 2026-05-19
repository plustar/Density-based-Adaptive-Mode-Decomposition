# damd/baselines/_admm.py
"""Shared ADMM solver for the VMD family.

A single implementation serves all three traditional baselines:

* ``dynamic=False``, single channel  →  **VMD**            (Dragomiretskiy & Zosso 2014)
* ``dynamic=False``, multi channel   →  **MVMD**           (Rehman & Aftab 2019)
* ``dynamic=True``,  any C           →  **dynamic STVMD**  (Jia et al. 2026)

The core update is the standard frequency-domain Wiener filter

.. math::
    \\hat u_k^{(n+1)} = \\frac{\\hat f - \\sum_{i\\ne k}\\hat u_i + \\hat\\lambda/2}
    {1 + 2\\alpha\\,(\\omega - \\omega_k)^2}

with a centre-frequency update that is either scalar (VMD/MVMD) or
per-frame (STVMD).

The implementation uses single-buffer Gauss–Seidel sweeps to avoid the
2× memory overhead of Jacobi-style updates.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

EPS = 1e-12


def admm_vmd(
    f_hat: np.ndarray,
    freqs: np.ndarray,
    K: int,
    *,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 500,
    omega_init: Optional[np.ndarray] = None,
    dynamic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """ADMM solver for the VMD family.

    Parameters
    ----------
    f_hat : (C, F, N) complex — Fourier transform of the (possibly
        windowed) input, with ``C`` channels, ``F`` frequency bins and
        ``N`` time points (=1 for plain VMD/MVMD; otherwise the number of
        STFT frames when running STVMD).
    freqs : (F,) normalised frequency axis in ``[0, 0.5]``
    K : number of modes
    alpha : bandwidth penalty weight
    tau : step size of the Lagrangian multiplier update (0 → no updates)
    tol, max_iter : convergence controls
    omega_init : (K,) or None — initial centre frequencies (uniformly
        spaced by default)
    dynamic : if True, ω varies per time point (STVMD)

    Returns
    -------
    u : (C, F, K, N) mode spectra
    omega : (K,) or (K, N) centre frequencies
    n_iter : iterations performed
    convergence : final change in ω
    """
    C, F, N = f_hat.shape

    # --- Initialise -------------------------------------------------
    if omega_init is not None:
        w = np.asarray(omega_init, dtype=float)
        if dynamic and w.ndim == 1:
            w = np.tile(w[:, None], (1, N))
    else:
        base = np.linspace(0.0, 0.5, K + 2)[1:-1]
        w = np.tile(base[:, None], (1, N)) if dynamic else base.copy()

    u = np.zeros((C, F, K, N), dtype=complex)
    lam = np.zeros((C, F, N), dtype=complex)
    w_prev = np.empty_like(w)
    convergence = np.inf
    n_iter = 0

    # --- ADMM iterations --------------------------------------------
    for it in range(max_iter):
        w_prev[:] = w

        # Mode updates (Gauss-Seidel)
        for k in range(K):
            if dynamic:
                dk = (freqs[:, None] - w[k, :][None, :]) ** 2
            else:
                dk = (freqs - w[k]) ** 2
                dk = dk[:, None]
            denom = 1.0 + 2.0 * alpha * dk
            other_sum = u.sum(axis=2) - u[:, :, k, :]
            num = f_hat - other_sum + lam / (2.0 * alpha + EPS)
            u[:, :, k, :] = num / (denom + EPS)

        # Centre-frequency update (eq. 5 / 50)
        power = (u.real ** 2 + u.imag ** 2).sum(axis=0)     # (F, K, N)
        if dynamic:
            num = (freqs[:, None, None] * power).sum(axis=0)  # (K, N)
            den = power.sum(axis=0)
        else:
            avg_power = power.mean(axis=2)                  # (F, K)
            num = (freqs[:, None] * avg_power).sum(axis=0)  # (K,)
            den = avg_power.sum(axis=0)
        safe = den > EPS
        w = np.where(safe, num / np.where(safe, den, 1.0), w_prev)

        # Lagrange multiplier
        if tau != 0.0:
            residual = f_hat - u.sum(axis=2)
            lam = lam + tau * residual

        convergence = float(np.sum((w - w_prev) ** 2) / max(w.size, 1))
        n_iter = it + 1
        if convergence < tol and it > 2:
            break

    # Sort modes by average frequency
    order = np.argsort(w.mean(axis=1) if dynamic else w)
    u = u[:, :, order, :]
    w = w[order] if not dynamic else w[order, :]
    return u, w, n_iter, convergence

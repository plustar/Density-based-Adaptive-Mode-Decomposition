# damd/baselines/stvmd.py
"""Short-Time Variational Mode Decomposition (Jia et al. 2026).

Sliding-window VMD with either shared centres (non-dynamic) or per-
frame centres (dynamic). Uses the shared ADMM core with ``dynamic``
forwarded to the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.windows import make_window
from ._admm import admm_vmd


@dataclass
class STVMDResult:
    """Output of STVMD.

    Notes
    -----
    ``modes`` has shape ``(C, K, N)`` after windowed overlap-add.
    ``centers`` is ``(K, T)`` in the dynamic case, ``(K,)`` otherwise.
    """

    modes: np.ndarray
    centers: np.ndarray
    n_iter: int
    convergence: float
    times: np.ndarray


def stvmd(
    X: np.ndarray,
    K: int,
    *,
    fs: float = 1.0,
    window_length: int = 256,
    hop_length: int = 1,
    window: str = "hann",
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    max_iter: int = 200,
    dynamic: bool = True,
    omega_init: Optional[np.ndarray] = None,
) -> STVMDResult:
    """Short-time (possibly dynamic) VMD."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    C, N = X.shape
    Lw = int(window_length)
    hop = int(hop_length)

    # Centre-pad so every sample has full frame coverage
    pad = Lw // 2
    Xp = np.pad(X, ((0, 0), (pad, pad)), mode="reflect")
    w = make_window(window, Lw)

    # Frame → (C, T, Lw)
    frames = np.lib.stride_tricks.sliding_window_view(
        Xp, Lw, axis=-1
    )[:, ::hop, :]
    T = frames.shape[1]
    windowed = frames * w[None, None, :]

    # rfft per frame → (C, T, F)
    Fw = np.fft.rfft(windowed, axis=-1)
    freqs = np.fft.rfftfreq(Lw)
    # ADMM wants (C, F, T)
    f_hat = np.transpose(Fw, (0, 2, 1))

    if omega_init is not None:
        omega_init = np.asarray(omega_init) / fs

    u, omega, n_iter, conv = admm_vmd(
        f_hat, freqs, K,
        alpha=alpha, tau=tau, tol=tol,
        max_iter=max_iter, omega_init=omega_init,
        dynamic=dynamic,
    )

    # --- Reconstruct time-domain modes via windowed overlap-add ---
    # u shape (C, F, K, T)
    u_ckft = np.transpose(u, (0, 2, 1, 3))          # (C, K, F, T)
    # irfft across F → (C, K, Lw, T)
    tdom = np.fft.irfft(u_ckft, n=Lw, axis=2)
    # Apply analysis window again for COLA
    tdom = tdom * w[None, None, :, None]

    out_len = Xp.shape[1]
    modes_full = np.zeros((C, K, out_len), dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)
    w2 = w ** 2
    for t in range(T):
        s = t * hop
        modes_full[:, :, s : s + Lw] += tdom[:, :, :, t]
        norm[s : s + Lw] += w2

    modes_full /= np.maximum(norm[None, None, :], 1e-10)
    modes = modes_full[:, :, pad : pad + N]
    times = (np.arange(T) * hop) / fs

    return STVMDResult(
        modes=modes,
        centers=omega * fs,
        n_iter=n_iter,
        convergence=conv,
        times=times,
    )

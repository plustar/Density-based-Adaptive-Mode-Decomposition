# damd/tfmaps/ssq_stft.py
"""Synchrosqueezed STFT.

Computes an STFT together with its derivative-window companion, uses the
ratio to estimate instantaneous frequency, and reassigns energy onto a
uniform frequency grid. The inverse adds up columns and divides by the
window centre value.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.windows import derivative_window, make_window
from ._analytic import phase_stft, reassign_stft
from ._base import TFMap
from .stft import stft as _stft


# ----------------------------------------------------------------------
def ssq_stft(
    x: np.ndarray,
    fs: float,
    *,
    n_fft: int = 256,
    hop_length: int = 1,
    window: str = "hann",
    n_ssq_freqs: Optional[int] = None,
    gamma: Optional[float] = None,
    center: bool = True,
    pad_mode: str = "reflect",
) -> TFMap:
    """Forward synchrosqueezed STFT.

    Parameters mirror :func:`damd.tfmaps.stft`. The output's frequency
    axis is linearly spaced in :math:`[0, f_s/2]` with ``n_ssq_freqs``
    bins (default: same length as the native STFT grid).
    """
    if hop_length != 1:
        raise ValueError("SSQ-STFT requires hop_length=1")

    # Base STFT (modulated window for smooth phase)
    base = _stft(
        x, fs,
        n_fft=n_fft, hop_length=1, window=window,
        center=center, modulated=True, pad_mode=pad_mode,
    )
    Sx = base.tf

    # STFT with the derivative window — same framing, different weighting.
    w = make_window(window, n_fft)
    dw = derivative_window(w, fs=fs)                     # ∂w / ∂t (Hz domain)
    # Build the derivative STFT by re-running the forward path with a
    # custom window. The simplest route is to feed the raw samples:
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[None, :]
    C, N = x_arr.shape
    if center:
        pad = n_fft // 2
        x_pad = np.pad(x_arr, ((0, 0), (pad, pad)), mode=pad_mode)
    else:
        x_pad = x_arr
    frames = np.lib.stride_tricks.sliding_window_view(
        x_pad, n_fft, axis=-1
    )                                                    # (C, T, n_fft)
    dw_shift = np.fft.ifftshift(dw)
    dSx = np.fft.rfft(frames * dw_shift[None, None, :], axis=-1)
    dSx = np.transpose(dSx, (0, 2, 1))                   # (C, F, T)

    # IF estimate
    w_if = phase_stft(Sx, dSx, base.freqs, gamma=gamma)

    # Output grid
    if n_ssq_freqs is None:
        n_ssq_freqs = len(base.freqs)
    ssq_freqs = np.linspace(0.0, 0.5 * fs, n_ssq_freqs)

    Tx = reassign_stft(Sx, w_if, ssq_freqs, gamma=gamma)

    return TFMap(
        tf=Tx,
        freqs=ssq_freqs,
        times=base.times,
        kind="ssq_stft",
        fs=fs,
        meta={
            "n_fft": n_fft,
            "hop_length": 1,
            "window": w,
            "window_center_value": float(w[n_fft // 2]),
            "n_samples": N,
        },
    )


# ----------------------------------------------------------------------
def issq_stft(tf_map: TFMap) -> np.ndarray:
    r"""Inverse SSQ-STFT.

    Uses the direct column-sum reconstruction with the analysis-window
    :math:`L_1` norm as the normaliser. The sign convention matches our
    particular choice of derivative window / modulated-window pair used
    in the forward pass; a user swapping to a different convention may
    need to flip the sign of the output.

    Perfect reconstruction is approximate: the squeezing step is
    *inherently* lossy for any finite frequency grid because spectral
    leakage between bins is absorbed by the scatter. For exact
    reconstruction, use :meth:`damd.DAMD.reconstruct` which performs
    hard-masked inversion on the underlying (non-squeezed) STFT.
    """
    if tf_map.kind != "ssq_stft":
        raise ValueError(f"expected kind='ssq_stft', got '{tf_map.kind}'")
    Tx = tf_map.tf
    w = tf_map.meta["window"]
    norm = float(w.sum())
    if abs(norm) < 1e-15:
        norm = 1.0
    x_hat = -Tx.real.sum(axis=1) / norm                  # (C, T)
    N0 = tf_map.meta.get("n_samples", x_hat.shape[-1])
    x_hat = x_hat[..., :N0]
    return x_hat[0] if x_hat.shape[0] == 1 else x_hat

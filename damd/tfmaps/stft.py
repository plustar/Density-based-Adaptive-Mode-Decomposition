# damd/tfmaps/stft.py
"""Short-Time Fourier Transform — forward and inverse.

Centre-padded STFT with exact overlap-add reconstruction at ``hop=1``
(used throughout the paper for perfect recovery). The forward pass
always returns a 3-D ``(C, F, T)`` complex map, even when the input is
1-D.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.freq_utils import rfft_freqs, stft_times
from ..core.windows import make_window
from ._base import TFMap


# ----------------------------------------------------------------------
def _frame(signal: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Zero-copy framing via :func:`numpy.lib.stride_tricks.sliding_window_view`."""
    frames = np.lib.stride_tricks.sliding_window_view(
        signal, n_fft, axis=-1
    )[..., ::hop, :]
    return frames


# ----------------------------------------------------------------------
def stft(
    x: np.ndarray,
    fs: float,
    *,
    n_fft: int = 256,
    hop_length: int = 1,
    window: str = "hann",
    center: bool = True,
    modulated: bool = True,
    pad_mode: str = "reflect",
) -> TFMap:
    """Forward short-time Fourier transform.

    Parameters
    ----------
    x : (N,) or (C, N) real signal
    fs : sampling rate in Hz
    n_fft : frame length
    hop_length : frame stride; ``1`` (the default) gives perfect-
        reconstruction overlap-add
    window : any name supported by :func:`scipy.signal.windows.get_window`
    center : if True, zero-pad by ``n_fft // 2`` on each side
    modulated : if True, ``ifftshift`` the window so that the zero-phase
        origin sits at the frame centre. Required for synchrosqueezing.

    Returns
    -------
    :class:`TFMap` of kind ``'stft'``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"expected 1-D or 2-D input, got shape {x.shape}")
    C, N = x.shape

    w = make_window(window, n_fft)
    if modulated:
        w_eff = np.fft.ifftshift(w)
    else:
        w_eff = w

    if center:
        pad = n_fft // 2
        x = np.pad(x, ((0, 0), (pad, pad)), mode=pad_mode)

    frames = _frame(x, n_fft, hop_length)           # (C, T, n_fft)
    windowed = frames * w_eff[None, None, :]
    S = np.fft.rfft(windowed, axis=-1)              # (C, T, F)
    tf = np.transpose(S, (0, 2, 1))                 # (C, F, T)

    freqs = rfft_freqs(n_fft, fs)
    times = stft_times(tf.shape[-1], hop_length, fs, n_fft, center=center)

    return TFMap(
        tf=tf,
        freqs=freqs,
        times=times,
        kind="stft",
        fs=fs,
        meta={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "window": w,                    # non-shifted, for iSTFT
            "center": center,
            "modulated": modulated,
            "n_samples": N,
        },
    )


# ----------------------------------------------------------------------
def istft(tf_map: TFMap, length: Optional[int] = None) -> np.ndarray:
    """Inverse STFT via weighted overlap-add.

    Reconstructs from a :class:`TFMap` produced by :func:`stft`. The
    implementation uses the squared analysis window as the synthesis
    weight (COLA-satisfied for Hann/Hamming at ``hop ≤ n_fft/2``).
    """
    if tf_map.kind != "stft":
        raise ValueError(f"expected kind='stft', got '{tf_map.kind}'")
    tf = tf_map.tf
    n_fft: int = tf_map.meta["n_fft"]
    hop: int = tf_map.meta["hop_length"]
    w: np.ndarray = tf_map.meta["window"]
    modulated: bool = tf_map.meta.get("modulated", True)
    center: bool = tf_map.meta.get("center", True)
    N0: int = tf_map.meta.get("n_samples", 0)

    C, F, T = tf.shape
    w_eff = np.fft.ifftshift(w) if modulated else w
    frames = np.fft.irfft(np.transpose(tf, (0, 2, 1)),
                          n=n_fft, axis=-1)         # (C, T, n_fft)

    out_len = (T - 1) * hop + n_fft
    output = np.zeros((C, out_len), dtype=np.float64)
    norm = np.zeros(out_len, dtype=np.float64)
    w2 = w_eff ** 2
    for t in range(T):
        s = t * hop
        output[:, s : s + n_fft] += frames[:, t, :] * w_eff
        norm[s : s + n_fft] += w2
    output /= np.maximum(norm, 1e-10)

    if center:
        pad = n_fft // 2
        output = output[:, pad:]
    if length is not None:
        output = output[:, :length]
    elif N0:
        output = output[:, :N0]
    # Squeeze if input was 1-D
    return output[0] if C == 1 else output

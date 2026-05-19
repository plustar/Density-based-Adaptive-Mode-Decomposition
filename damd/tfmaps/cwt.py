# damd/tfmaps/cwt.py
"""Continuous Wavelet Transform.

Batched FFT-convolution implementation: transform the signal once, build
the full wavelet bank in one vectorised call, multiply, and single
inverse FFT over all scales simultaneously.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.freq_utils import log_scales, scales_to_freqs
from ._base import TFMap
from ._wavelets import (
    admissibility_constant,
    get_wavelet,
    wavelet_center_freq,
)


# ----------------------------------------------------------------------
def _auto_n_scales(n_samples: int) -> int:
    """Default number of scales — matches ssqueezepy's ``nv=32`` rule."""
    voices = 32
    return max(16, int(voices * np.log2(max(n_samples, 16))))


def _auto_scale_range(
    n_samples: int,
    fs: float,
    center_freq: float,
) -> Tuple[float, float]:
    """Default frequency range: one cycle in the full signal → Nyquist."""
    f_min = max(fs / n_samples, 1e-3)
    f_max = 0.5 * fs
    return f_min, f_max


# ----------------------------------------------------------------------
def cwt(
    x: np.ndarray,
    fs: float,
    *,
    wavelet: str = "gmw",
    wavelet_params: Optional[dict] = None,
    n_scales: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> TFMap:
    """Forward continuous wavelet transform.

    Parameters
    ----------
    x : (N,) or (C, N) real signal
    fs : sampling rate
    wavelet : ``'morlet'`` | ``'gmw'`` | ``'bump'``
    n_scales : number of log-spaced scales (default ≈ 32 per octave)
    f_min, f_max : frequency band in Hz (defaults: ``fs/N`` to ``fs/2``)

    Returns
    -------
    :class:`TFMap` of kind ``'cwt'``. Frequencies increase along axis 1.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    C, N = x.shape

    params = wavelet_params or {}
    omega0 = wavelet_center_freq(wavelet, params)
    if n_scales is None:
        n_scales = _auto_n_scales(N)
    if f_min is None or f_max is None:
        fmn, fmx = _auto_scale_range(N, fs, omega0)
        f_min = f_min or fmn
        f_max = f_max or fmx
    scales = log_scales(n_scales, f_min, f_max, fs, omega0)     # ascending → f desc

    # Pad to next power of two for FFT efficiency
    N_up = 1 << (max(N, 2) - 1).bit_length()
    pad_l = (N_up - N) // 2
    pad_r = N_up - N - pad_l
    if N_up > N:
        x_pad = np.pad(x, ((0, 0), (pad_l, pad_r)), mode="reflect")
    else:
        x_pad = x

    # ξ in rad/sample, one-sided FFT grid (with negative frequencies)
    xi = np.fft.fftfreq(N_up) * 2.0 * np.pi                    # (N_up,)
    psi_fn = get_wavelet(wavelet, params)
    # Batched wavelet bank: conj for convolution, sqrt(scale) keeps L2-norm
    Psi = np.conj(psi_fn(scales[:, None] * xi[None, :])) * np.sqrt(scales)[:, None]

    X = np.fft.fft(x_pad, axis=-1)                             # (C, N_up)
    prod = X[:, None, :] * Psi[None, :, :]                     # (C, S, N_up)
    tf = np.fft.ifft(prod, axis=-1)
    if N_up > N:
        tf = tf[:, :, pad_l : pad_l + N]

    # Reverse so that freqs[] is ascending
    tf = tf[:, ::-1, :]
    scales_asc_f = scales[::-1]
    freqs = scales_to_freqs(scales_asc_f, fs, omega0)          # Hz, ascending
    times = np.arange(N) / fs

    return TFMap(
        tf=tf,
        freqs=freqs,
        times=times,
        kind="cwt",
        fs=fs,
        meta={
            "wavelet": wavelet,
            "wavelet_params": params,
            "scales": scales_asc_f,
            "omega0": omega0,
            "n_samples": N,
        },
    )


# ----------------------------------------------------------------------
def icwt(tf_map: TFMap) -> np.ndarray:
    """Inverse CWT via admissibility-weighted integration.

    Uses the standard reconstruction formula :math:`x(t) = C_\\psi^{-1}
    \\int_0^\\infty \\text{Re}\\,W(a,t)\\, a^{-3/2}\\, da`. Perfect only
    in the limit of a dense scale grid; for the paper's experiments the
    default 32-voice grid yields RMSE of order 10⁻².
    """
    if tf_map.kind != "cwt":
        raise ValueError(f"expected kind='cwt', got '{tf_map.kind}'")
    tf = tf_map.tf                                    # (C, S, T)
    scales = tf_map.meta["scales"]                    # ascending in frequency
    wavelet = tf_map.meta["wavelet"]
    params = tf_map.meta.get("wavelet_params", {})

    # da in log-scale units; the factor scales^{-3/2} comes from the CWT
    # admissibility relation when the wavelet is L^2-normalised.
    log_s = np.log(scales)
    da = np.empty_like(scales)
    da[1:-1] = 0.5 * (log_s[2:] - log_s[:-2])
    da[0] = log_s[1] - log_s[0]
    da[-1] = log_s[-1] - log_s[-2]
    weights = np.abs(da) / scales ** 1.5

    C_psi = admissibility_constant(wavelet, params)
    C_psi = max(C_psi, 1e-10)

    recon = np.real(np.sum(tf * weights[None, :, None], axis=1)) / C_psi
    C = recon.shape[0]
    return recon[0] if C == 1 else recon

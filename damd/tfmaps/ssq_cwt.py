# damd/tfmaps/ssq_cwt.py
"""Synchrosqueezed CWT.

Computes a CWT together with its derivative-wavelet companion and
reassigns energy to instantaneous-frequency bins. Inverse follows the
standard :math:`2/C_\\psi` formula.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..core.freq_utils import log_scales, scales_to_freqs
from ._analytic import phase_cwt, reassign_cwt
from ._base import TFMap
from ._wavelets import (
    admissibility_constant,
    get_wavelet,
    wavelet_center_freq,
)
from .cwt import _auto_n_scales, _auto_scale_range


# ----------------------------------------------------------------------
def _cwt_pair(
    x: np.ndarray,
    scales: np.ndarray,
    wavelet: str,
    params: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Joint forward CWT and derivative-wavelet CWT.

    Uses ``ψ̂'(ω) = i ω ψ̂(ω)`` in the frequency domain — cheap because
    the wavelet is already in frequency form.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    C, N = x.shape
    N_up = 1 << (max(N, 2) - 1).bit_length()
    pad_l = (N_up - N) // 2
    pad_r = N_up - N - pad_l
    if N_up > N:
        x = np.pad(x, ((0, 0), (pad_l, pad_r)), mode="reflect")

    xi = np.fft.fftfreq(N_up) * 2.0 * np.pi              # rad / sample
    psi_fn = get_wavelet(wavelet, params)
    Psi = np.conj(psi_fn(scales[:, None] * xi[None, :])) * np.sqrt(scales)[:, None]
    # Derivative wavelet: multiply by i*ω in frequency (note: conjugate
    # already applied, so use -i*ω)
    dPsi = Psi * (1j * xi[None, :])

    X = np.fft.fft(x, axis=-1)
    Wx = np.fft.ifft(X[:, None, :] * Psi[None, :, :], axis=-1)
    dWx = np.fft.ifft(X[:, None, :] * dPsi[None, :, :], axis=-1)
    if N_up > N:
        Wx = Wx[:, :, pad_l : pad_l + N]
        dWx = dWx[:, :, pad_l : pad_l + N]
    return Wx, dWx


# ----------------------------------------------------------------------
def ssq_cwt(
    x: np.ndarray,
    fs: float,
    *,
    wavelet: str = "gmw",
    wavelet_params: Optional[dict] = None,
    n_scales: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    n_ssq_freqs: Optional[int] = None,
    gamma: Optional[float] = None,
) -> TFMap:
    """Forward synchrosqueezed CWT."""
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
    scales = log_scales(n_scales, f_min, f_max, fs, omega0)

    Wx, dWx = _cwt_pair(x, scales, wavelet, params)
    # Reverse so that the native frequency axis is ascending
    Wx = Wx[:, ::-1, :]
    dWx = dWx[:, ::-1, :]
    scales_asc = scales[::-1]
    native_freqs = scales_to_freqs(scales_asc, fs, omega0)  # ascending

    # Phase transform (IF estimate at every scale/time)
    w_if = phase_cwt(Wx, dWx, gamma=gamma)

    # Output grid
    if n_ssq_freqs is None:
        n_ssq_freqs = n_scales
    ssq_freqs = np.geomspace(native_freqs[0], native_freqs[-1], n_ssq_freqs)

    Tx = reassign_cwt(Wx, w_if, ssq_freqs, scales_asc, gamma=gamma)
    times = np.arange(N) / fs

    return TFMap(
        tf=Tx,
        freqs=ssq_freqs,
        times=times,
        kind="ssq_cwt",
        fs=fs,
        meta={
            "wavelet": wavelet,
            "wavelet_params": params,
            "scales": scales_asc,
            "omega0": omega0,
            "n_samples": N,
        },
    )


# ----------------------------------------------------------------------
def issq_cwt(tf_map: TFMap) -> np.ndarray:
    r"""Inverse SSQ-CWT: :math:`x(t) = \frac{2}{C_\psi}\operatorname{Re}
    \sum_k T_x(f_k, t)`."""
    if tf_map.kind != "ssq_cwt":
        raise ValueError(f"expected kind='ssq_cwt', got '{tf_map.kind}'")
    wavelet = tf_map.meta["wavelet"]
    params = tf_map.meta.get("wavelet_params", {})
    C_psi = max(admissibility_constant(wavelet, params), 1e-10)
    x_hat = tf_map.tf.real.sum(axis=1) * (2.0 / C_psi)      # (C, T)
    N0 = tf_map.meta.get("n_samples", x_hat.shape[-1])
    x_hat = x_hat[..., :N0]
    return x_hat[0] if x_hat.shape[0] == 1 else x_hat

# damd/core/freq_utils.py
"""Frequency-axis helpers.

All transforms in this package expose a ``freqs`` array in Hz. These
utilities convert between bin indices, normalised frequency, and Hz.
"""
from __future__ import annotations

import numpy as np


def rfft_freqs(n_fft: int, fs: float) -> np.ndarray:
    """One-sided STFT bin frequencies in Hz."""
    return np.fft.rfftfreq(n_fft, d=1.0 / fs).astype(np.float64)


def log_scales(
    n_scales: int,
    f_min: float,
    f_max: float,
    fs: float,
    wavelet_center_freq: float,
) -> np.ndarray:
    """Log-spaced CWT scales with prescribed frequency support.

    For a wavelet with centre (angular) frequency ``ω0`` the instantaneous
    frequency at scale ``a`` is ``f = fs * ω0 / (2πa)``. Inverting
    yields scales that sample ``[f_min, f_max]`` geometrically.
    """
    omega0 = 2.0 * np.pi * wavelet_center_freq     # rad / sample
    # f = fs * ω0 / (2π a)  ⇒  a = fs * ω0 / (2π f)
    a_max = fs * omega0 / (2.0 * np.pi * f_min)
    a_min = fs * omega0 / (2.0 * np.pi * f_max)
    return np.geomspace(a_min, a_max, n_scales)


def scales_to_freqs(
    scales: np.ndarray,
    fs: float,
    wavelet_center_freq: float,
) -> np.ndarray:
    """Frequency in Hz corresponding to each CWT scale."""
    omega0 = 2.0 * np.pi * wavelet_center_freq
    return fs * omega0 / (2.0 * np.pi * scales)


def stft_times(
    n_frames: int,
    hop_length: int,
    fs: float,
    n_fft: int,
    center: bool = True,
) -> np.ndarray:
    """Time axis of an STFT with centre padding."""
    if center:
        return np.arange(n_frames) * hop_length / fs
    return (np.arange(n_frames) * hop_length + n_fft / 2) / fs

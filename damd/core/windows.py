# damd/core/windows.py
"""Window functions for STFT.

Wraps :func:`scipy.signal.windows.get_window` with a couple of defaults
used by the paper experiments.
"""
from __future__ import annotations

import numpy as np
from scipy.signal.windows import get_window as _scipy_window


def make_window(name: str | tuple, n: int) -> np.ndarray:
    """Build a length-``n`` window by name.

    Accepts any window string supported by :mod:`scipy.signal.windows`,
    plus the common aliases ``'hann'``, ``'hamming'``, ``'blackman'``,
    ``'boxcar'``.
    """
    if isinstance(name, str) and name.lower() == "rect":
        name = "boxcar"
    return _scipy_window(name, n, fftbins=False).astype(np.float64)


def derivative_window(window: np.ndarray, fs: float = 1.0) -> np.ndarray:
    """Numerical derivative of a window (for SSQ-STFT phase transform).

    Uses central differences with edge reflection.
    """
    w = np.asarray(window, dtype=np.float64)
    dw = np.empty_like(w)
    dw[1:-1] = (w[2:] - w[:-2]) / 2.0
    dw[0] = w[1] - w[0]
    dw[-1] = w[-1] - w[-2]
    return dw * fs

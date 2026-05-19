# damd/tfmaps/_analytic.py
"""Synchrosqueezing primitives shared by SSQ-STFT and SSQ-CWT.

Algorithm follows Daubechies–Lu–Wu (ACHA 2011) and Thakur–Wu,
matching the reference implementation in ``ssqueezepy``:

* phase transform — instantaneous-frequency estimate from the ratio of
  the transform with the derivative wavelet/window to the transform
  itself
* reassignment   — scatter energy into the IF-aligned bin, with ``da``
  weighting for the CWT case
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

EPS64 = np.finfo(np.float64).eps


def _gamma_default(arr: np.ndarray) -> float:
    """Conservative noise threshold — 10× machine epsilon by convention."""
    return 10.0 * (
        EPS64 if arr.dtype in (np.complex128, np.float64)
        else np.finfo(np.float32).eps
    )


# ----------------------------------------------------------------------
# Phase transforms
# ----------------------------------------------------------------------
def phase_cwt(Wx: np.ndarray, dWx: np.ndarray,
              gamma: float | None = None) -> np.ndarray:
    r"""IF estimate for CWT coefficients.

    .. math::
        w[a,b] = \left|\frac{-\text{Im}\left(dW_x/W_x\right)}{2\pi}\right|

    Entries with ``|Wx| < gamma`` are set to ``+inf`` and later skipped
    by the reassignment step.
    """
    if gamma is None:
        gamma = _gamma_default(Wx)
    w = np.full(Wx.shape, np.inf, dtype=Wx.real.dtype)
    mask = np.abs(Wx) >= gamma
    if mask.any():
        w[mask] = np.abs(-np.imag(dWx[mask] / Wx[mask]) / (2.0 * np.pi))
    return w


def phase_stft(Sx: np.ndarray, dSx: np.ndarray,
               freqs_hz: np.ndarray,
               gamma: float | None = None) -> np.ndarray:
    r"""IF estimate for STFT coefficients.

    .. math::
        w[u,k] = f_k - \frac{\text{Im}\left(dS_x/S_x\right)}{2\pi}

    with the same ``|Sx| < gamma`` masking.
    """
    if gamma is None:
        gamma = _gamma_default(Sx)
    w = np.full(Sx.shape, np.inf, dtype=Sx.real.dtype)
    mask = np.abs(Sx) >= gamma
    if mask.any():
        ratio = np.zeros_like(Sx, dtype=complex)
        ratio[mask] = dSx[mask] / Sx[mask]
        # Broadcast frequency axis into (C, F, T)
        f_bc = freqs_hz[None, :, None] if Sx.ndim == 3 else freqs_hz[:, None]
        w_full = np.broadcast_to(f_bc, Sx.shape).astype(Sx.real.dtype).copy()
        w_full -= np.imag(ratio) / (2.0 * np.pi)
        w[mask] = w_full[mask]
    return w


# ----------------------------------------------------------------------
# Reassignment
# ----------------------------------------------------------------------
def _scatter_add(Tx: np.ndarray, coef: np.ndarray,
                 target: np.ndarray, ok_mask: np.ndarray) -> None:
    """In-place scatter accumulation: Tx[c, target[c,i,t], t] += coef[c,i,t]."""
    if not ok_mask.any():
        return
    C, _, T = Tx.shape
    F_out = Tx.shape[1]
    ch_idx = np.broadcast_to(np.arange(C)[:, None, None], ok_mask.shape)
    t_idx = np.broadcast_to(np.arange(T)[None, None, :], ok_mask.shape)

    flat = ch_idx[ok_mask] * (F_out * T) + target[ok_mask] * T + t_idx[ok_mask]
    np.add.at(Tx.ravel(), flat, coef[ok_mask])


def reassign_cwt(Wx: np.ndarray, w: np.ndarray,
                 ssq_freqs: np.ndarray,
                 scales: np.ndarray,
                 gamma: float | None = None) -> np.ndarray:
    """Synchrosqueeze CWT coefficients onto ``ssq_freqs`` (in Hz).

    Includes the log-scale ``da`` correction that turns the native CWT
    integration variable into :math:`d(\\log a)`.
    """
    if gamma is None:
        gamma = _gamma_default(Wx)
    if Wx.ndim != 3:
        raise ValueError("Wx must be (C, S, T)")
    C, S, T = Wx.shape
    F = len(ssq_freqs)
    Tx = np.zeros((C, F, T), dtype=Wx.dtype)

    if S > 1:
        log_s = np.log(scales)
        da = np.empty(S)
        da[0] = log_s[1] - log_s[0]
        da[-1] = log_s[-1] - log_s[-2]
        da[1:-1] = 0.5 * (log_s[2:] - log_s[:-2])
        da = np.abs(da * scales)
    else:
        da = np.ones(S)

    Wx_da = Wx * da[None, :, None]
    target = np.searchsorted(ssq_freqs, w.ravel()).reshape(C, S, T)
    target = np.clip(target, 0, F - 1)
    ok = np.isfinite(w) & (np.abs(Wx) >= gamma)
    _scatter_add(Tx, Wx_da, target, ok)
    return Tx


def reassign_stft(Sx: np.ndarray, w: np.ndarray,
                  ssq_freqs: np.ndarray,
                  gamma: float | None = None) -> np.ndarray:
    """Synchrosqueeze STFT coefficients onto ``ssq_freqs`` (in Hz)."""
    if gamma is None:
        gamma = _gamma_default(Sx)
    if Sx.ndim != 3:
        raise ValueError("Sx must be (C, F, T)")
    C, F_in, T = Sx.shape
    F_out = len(ssq_freqs)
    Tx = np.zeros((C, F_out, T), dtype=Sx.dtype)

    target = np.searchsorted(ssq_freqs, w.ravel()).reshape(C, F_in, T)
    target = np.clip(target, 0, F_out - 1)
    ok = np.isfinite(w) & (np.abs(Sx) >= gamma)
    _scatter_add(Tx, Sx, target, ok)
    return Tx

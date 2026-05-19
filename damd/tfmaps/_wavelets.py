# damd/tfmaps/_wavelets.py
"""Frequency-domain analytic wavelet formulas, NumPy only.

Three wavelet families are provided — Morlet, the Generalised Morse
Wavelet (GMW), and Bump. Each is expressed analytically in the
frequency domain so that the CWT reduces to an FFT-domain
multiplication.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np


# ----------------------------------------------------------------------
def morlet_freq(xi: np.ndarray, mu: float = 6.0) -> np.ndarray:
    r"""Morlet wavelet: :math:`\Psi(\xi) = \pi^{-1/4} e^{-\frac{1}{2}(\xi-\mu)^2}`.

    ``mu`` is the dimensionless centre frequency; 6.0 is the usual choice
    that keeps the wavelet approximately analytic.
    """
    return math.pi ** (-0.25) * np.exp(-0.5 * (xi - mu) ** 2)


def gmw_freq(xi: np.ndarray, beta: float = 3.0,
             gamma: float = 3.0) -> np.ndarray:
    r"""Generalised Morse Wavelet.

    :math:`\Psi(\xi) = a\, \xi^{\beta} e^{-\gamma \xi^{\gamma}}\,\mathbb{1}_{\xi\ge 0}`
    with :math:`a = 2 (e\gamma/\beta)^{\beta/\gamma}` (peak-normalised).
    """
    a = 2.0 * (math.e * gamma / beta) ** (beta / gamma)
    xi_pos = np.maximum(xi, 0.0)
    heaviside = (xi >= 0).astype(xi.dtype)
    return a * xi_pos ** beta * np.exp(-gamma * xi_pos ** gamma) * heaviside


def bump_freq(xi: np.ndarray, sigma: float = 0.6,
              mu: float = 1.0) -> np.ndarray:
    r"""Bump wavelet with compact frequency support.

    :math:`\Psi(\xi) = \exp\!\left(1 - \frac{1}{1-t^2}\right)\,\mathbb{1}_{|t|<1}`,
    where :math:`t = (\xi - \mu)/\sigma`.
    """
    t = (xi - mu) / sigma
    t_sq = t ** 2
    inside = t_sq < 1.0
    safe = np.where(inside, t_sq, 0.5)
    vals = np.exp(-1.0 / (1.0 - safe) + 1.0)
    return np.where(inside, vals, 0.0)


# ----------------------------------------------------------------------
_WAVELETS = {
    "morlet": (morlet_freq, {"mu": 6.0}),
    "gmw":    (gmw_freq,    {"beta": 3.0, "gamma": 3.0}),
    "bump":   (bump_freq,   {"sigma": 0.6, "mu": 1.0}),
}


def get_wavelet(name: str, params: Optional[dict] = None
                ) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable ``ψ̂(ξ)`` for one of the supported wavelets."""
    name = name.lower()
    if name not in _WAVELETS:
        raise ValueError(
            f"unknown wavelet '{name}'. Supported: {list(_WAVELETS)}"
        )
    fn, defaults = _WAVELETS[name]
    merged = {**defaults, **(params or {})}
    return lambda xi, _fn=fn, _kw=merged: _fn(xi, **_kw)


def wavelet_center_freq(name: str,
                        params: Optional[dict] = None) -> float:
    """Centre frequency :math:`\\omega_0` (rad/sample → cycles/sample) of the
    given wavelet, used to map scales to Hz.

    We compute it numerically as the argmax of :math:`|\\hat\\psi(\\xi)|`
    on a fine grid — robust for any analytic wavelet.
    """
    fn = get_wavelet(name, params)
    xi = np.linspace(0.0, 20.0, 10_001)
    mag = np.abs(fn(xi))
    peak = xi[int(np.argmax(mag))]
    # Return cycles / sample (divide by 2π)
    return peak / (2.0 * np.pi)


def admissibility_constant(name: str,
                           params: Optional[dict] = None) -> float:
    """:math:`C_\\psi = \\int_0^\\infty \\hat\\psi(\\xi)/\\xi\\, d\\xi`.

    Needed for the inverse SSQ-CWT reconstruction (eq. analogous to
    eq. 25 of Daubechies–Lu–Wu 2011). Computed by trapezoid rule on a
    dense log grid.
    """
    fn = get_wavelet(name, params)
    xi = np.geomspace(1e-4, 50.0, 20_000)
    integrand = fn(xi) / xi
    integrand = np.real(integrand)
    return float(np.trapezoid(integrand, xi))

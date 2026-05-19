# experiments/common/signals.py
"""Reference signal generators matching the two papers' §III / §VIII
experimental setups.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ======================================================================
# DAMD paper §III-A
# ======================================================================
def noisy_sinusoid(
    fs: float = 512.0,
    n_samples: int = 1024,
    f0: float = 50.0,
    amplitude: float = 0.2,
    noise_std: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Noisy sinusoid, eq. (61) of the DAMD paper.

    ``x(t) = As sin(2π f₀ t) + σₙ w(t)`` with ``w ~ N(0, 1)``.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    x = amplitude * np.sin(2 * np.pi * f0 * t) + noise_std * rng.standard_normal(n_samples)
    return x, t


def simulated_signal(
    fs: float = 128.0,
    n_samples: int = 1024,
    noise_std: float = 0.04,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Multi-component signal with time-varying frequency, harmonics and
    an intermittent burst — eq. (62)–(66) of the DAMD paper.

    Returns
    -------
    x : (N,) signal
    t : (N,) time axis
    truth : dict describing each component (for validation)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    # Random permutation of integers 0..7, stepped per second
    seq = rng.permutation(8)
    omega = seq[np.floor(t).astype(int) % len(seq)] + 13.0     # base 13 Hz
    x1 = np.sin(2 * np.pi * omega * t)
    x2 = 0.75 * np.sin(2 * np.pi * 1.75 * omega * t)
    # Binary mask for intermittent third component
    mask = ((t >= 1.5) & (t <= 3.5)) | ((t >= 5.5) & (t <= 7.5))
    x3 = 1.25 * np.sin(2 * np.pi * 2.5 * omega * t) * mask
    noise = noise_std * rng.standard_normal(n_samples)
    x = x1 + x2 + x3 + noise
    return x, t, {"omega": omega, "mask": mask,
                  "components": ("fundamental", "1.75×", "2.5×"),}


# ======================================================================
# MDAMD paper §VIII-A Signal 1 (d=4)
# ======================================================================
def multichannel_signal(
    fs: float = 256.0,
    duration: float = 4.0,
    n_channels: int = 4,
    noise_std: float = 0.3,
    shared_freq: float = 50.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Multi-channel signal with one coherent shared mode.

    Per-channel: three channel-specific components + shared ``f_shared``
    in-phase across channels + additive Gaussian noise. This matches
    the setup of §VIII-A / Fig 1 of the MDAMD paper, simplified (we
    drop the FM / chirp / burst components for brevity).
    """
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    shared = np.sin(2 * np.pi * shared_freq * t)

    # Channel-specific distinct components
    channel_freqs = [20, 35, 60, 80][:n_channels]
    specifics = np.vstack([
        np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in channel_freqs
    ])
    X = specifics + shared[None, :] + noise_std * rng.standard_normal((n_channels, N))
    return X, t, {"shared_freq": shared_freq,
                  "channel_freqs": channel_freqs}


# ======================================================================
# MDAMD paper §VIII-A Signal 2 (d=100, high-dimensional)
# ======================================================================
def high_dim_signal(
    fs: float = 256.0,
    n_samples: int = 1024,
    n_channels: int = 100,
    n_modes: int = 4,
    mode_freqs: Optional[List[float]] = None,
    noise_std: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """High-dimensional low-rank signal.

    Energy concentrates in a ``n_modes``-dimensional subspace while
    noise distributes isotropically — the setting in which projection-
    based aggregation dominates direct aggregation (Prop 5 of MDAMD).
    """
    rng = np.random.default_rng(seed)
    freqs = mode_freqs or [15.0, 30.0, 50.0, 80.0][:n_modes]
    t = np.arange(n_samples) / fs
    # Orthonormal mixing basis
    U, _ = np.linalg.qr(rng.standard_normal((n_channels, n_modes)))
    sources = np.vstack([np.sin(2 * np.pi * f * t) for f in freqs])
    X = U @ sources + noise_std * rng.standard_normal((n_channels, n_samples))
    return X, t, {"U_true": U, "mode_freqs": freqs}

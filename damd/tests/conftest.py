# damd/tests/conftest.py
"""Shared test fixtures and signal generators."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def noisy_sinusoid(rng):
    """DAMD paper §III-A noisy sinusoid: 50 Hz + white noise."""
    fs, N = 512, 1024
    t = np.arange(N) / fs
    x = 0.2 * np.sin(2 * np.pi * 50 * t) + 0.5 * rng.standard_normal(N)
    return x, fs, 50.0


@pytest.fixture(scope="session")
def multichannel_coherent(rng):
    """Multivariate test signal: shared 50 Hz across 4 channels with noise."""
    fs, N, d = 256, 1024, 4
    t = np.arange(N) / fs
    shared = np.sin(2 * np.pi * 50 * t)
    X = np.vstack([shared + 0.3 * rng.standard_normal(N) for _ in range(d)])
    return X, fs, 50.0


@pytest.fixture(scope="session")
def high_dim(rng):
    """d=50, r=4 low-rank signal at 15/30/50/80 Hz with isotropic noise."""
    fs, N, d, r = 256, 1024, 50, 4
    t = np.arange(N) / fs
    truth = [15.0, 30.0, 50.0, 80.0]
    U_true, _ = np.linalg.qr(rng.standard_normal((d, r)))
    sources = np.vstack([np.sin(2 * np.pi * f * t) for f in truth])
    X = U_true @ sources + 0.5 * rng.standard_normal((d, N))
    return X, fs, truth, U_true


@pytest.fixture(scope="session")
def two_tone(rng):
    """Clean 50 Hz + 100 Hz, mild noise — good for VMD K=2 tests."""
    fs, N = 512, 1024
    t = np.arange(N) / fs
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t) + \
        0.1 * rng.standard_normal(N)
    return x, fs, [50.0, 100.0]

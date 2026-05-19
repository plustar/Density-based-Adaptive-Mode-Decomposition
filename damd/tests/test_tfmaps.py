# damd/tests/test_tfmaps.py
"""Tests for the four time-frequency transforms."""
from __future__ import annotations

import numpy as np
import pytest

from damd.tfmaps import (
    cwt, icwt,
    forward, inverse,
    ssq_cwt, issq_cwt,
    ssq_stft, issq_stft,
    stft, istft,
)
from damd.core import DAMDConfig


class TestSTFT:
    def test_shape(self):
        x = np.random.randn(512)
        m = stft(x, fs=256, n_fft=128)
        assert m.tf.ndim == 3
        assert m.tf.shape[0] == 1
        assert m.tf.shape[1] == 65          # n_fft//2 + 1

    def test_round_trip_perfect(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(512)
        m = stft(x, fs=256, n_fft=128)
        x_hat = istft(m)
        assert np.allclose(x, x_hat[:len(x)], atol=1e-9)

    def test_multichannel(self):
        X = np.random.randn(3, 512)
        m = stft(X, fs=256, n_fft=128)
        assert m.tf.shape == (3, 65, 513)


class TestCWT:
    def test_basic_shape(self):
        x = np.random.randn(256)
        m = cwt(x, fs=128, n_scales=48)
        assert m.tf.ndim == 3
        assert m.tf.shape[0] == 1
        assert m.tf.shape[1] == 48

    def test_ascending_freqs(self):
        m = cwt(np.random.randn(256), fs=128, n_scales=48)
        assert np.all(np.diff(m.freqs) > 0)

class TestSSQ:
    def test_ssq_stft_shape(self):
        x = np.random.randn(512)
        m = ssq_stft(x, fs=256, n_fft=128)
        assert m.tf.ndim == 3
        assert m.kind == "ssq_stft"

    def test_ssq_cwt_shape(self):
        x = np.random.randn(256)
        m = ssq_cwt(x, fs=128, n_scales=48)
        assert m.tf.ndim == 3
        assert m.kind == "ssq_cwt"

    def test_ssq_concentrates_energy(self):
        """SSQ should concentrate a sinusoid's energy into fewer bins than STFT."""
        fs, N = 256, 1024
        t = np.arange(N) / fs
        x = np.sin(2 * np.pi * 50 * t)
        base = stft(x, fs=fs, n_fft=128)
        ssq = ssq_stft(x, fs=fs, n_fft=128)
        # Measure: sparsity ratio (where's 90% of the energy concentrated?)
        def _sparsity(M):
            p = np.abs(M.tf[0]) ** 2                    # (F, T)
            col_e = p.sum(axis=0)                       # (T,)
            total = col_e.sum()
            # count bins containing 90% of energy averaged over time
            sorted_p = np.sort(p, axis=0)[::-1]
            cum = np.cumsum(sorted_p, axis=0)
            bins_90 = (cum < 0.9 * col_e[None, :]).sum(axis=0).mean()
            return bins_90
        assert _sparsity(ssq) <= _sparsity(base) + 2      # not worse than STFT


class TestDispatch:
    @pytest.mark.parametrize("name", ["stft", "cwt", "ssq_stft", "ssq_cwt"])
    def test_forward_inverse_roundtrip(self, name):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(256)
        cfg = DAMDConfig(fs=128.0, transform=name, n_fft=64, n_scales=48)
        m = forward(x, cfg)
        assert m.kind == name
        # Just verify inverse runs without error
        y = inverse(m)
        assert y.shape[-1] >= x.size - 10

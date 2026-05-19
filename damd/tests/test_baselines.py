# damd/tests/test_baselines.py
"""Tests for the VMD / MVMD / STVMD baselines."""
from __future__ import annotations

import numpy as np
import pytest

from damd.baselines import mvmd, stvmd, vmd


class TestVMD:
    def test_recovers_two_tones(self, two_tone):
        x, fs, truth = two_tone
        res = vmd(x, K=2, fs=fs, alpha=2000)
        centers = sorted(res.centers)
        assert abs(centers[0] - truth[0]) < 1.0
        assert abs(centers[1] - truth[1]) < 1.0

    def test_reconstruction_error_bounded_by_noise(self, two_tone):
        x, fs, _ = two_tone
        res = vmd(x, K=2, fs=fs, alpha=2000)
        rmse = np.sqrt(np.mean((x - res.modes.sum(0)) ** 2))
        # Two clean tones + 0.1·noise → residual ≈ 0.1
        assert rmse < 0.15

    def test_convergence_reported(self, two_tone):
        x, fs, _ = two_tone
        res = vmd(x, K=2, fs=fs, max_iter=100)
        assert res.n_iter > 0
        assert res.convergence >= 0


class TestMVMD:
    def test_shared_centers(self, two_tone, rng):
        x, fs, truth = two_tone
        X = np.vstack([x + 0.05 * rng.standard_normal(x.size) for _ in range(3)])
        res = mvmd(X, K=2, fs=fs)
        centers = sorted(res.centers)
        assert abs(centers[0] - truth[0]) < 1.0
        assert abs(centers[1] - truth[1]) < 1.0
        # Centers are shared across channels — single (K,) vector
        assert res.centers.ndim == 1
        assert res.modes.shape[0] == 3

    def test_handles_single_channel(self, two_tone):
        """MVMD with d=1 should behave identically to VMD."""
        x, fs, truth = two_tone
        res = mvmd(x[None, :], K=2, fs=fs)
        assert res.modes.shape[0] == 1
        centers = sorted(res.centers)
        assert abs(centers[0] - truth[0]) < 1.0


class TestSTVMD:
    def test_dynamic_center_shape(self, two_tone):
        x, fs, _ = two_tone
        res = stvmd(x[:512], K=2, fs=fs,
                    window_length=128, hop_length=16, max_iter=30, dynamic=True)
        # Centers per frame
        assert res.centers.ndim == 2
        assert res.centers.shape[0] == 2

    def test_non_dynamic_shares_center(self, two_tone):
        x, fs, _ = two_tone
        res = stvmd(x[:512], K=2, fs=fs,
                    window_length=128, hop_length=16, max_iter=30, dynamic=False)
        assert res.centers.ndim == 1

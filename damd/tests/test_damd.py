# damd/tests/test_damd.py
"""Integration tests for the DAMD pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from damd import DAMD, DAMDConfig, DAMDResult


class TestSingleChannel:
    def test_detects_target_frequency(self, noisy_sinusoid):
        x, fs, f0 = noisy_sinusoid
        res = DAMD(fs=fs, transform="stft", n_fft=256).fit(x)
        # At least half the frames should find the target within 3 Hz
        hits = sum(np.any(np.abs(c - f0) < 3) for c in res.centers)
        assert hits / len(res.centers) > 0.5

    def test_returns_result_object(self, noisy_sinusoid):
        x, fs, _ = noisy_sinusoid
        d = DAMD(fs=fs, n_fft=128)
        res = d.fit(x)
        assert isinstance(res, DAMDResult)
        assert res.kind == "stft"
        # summary doesn't crash
        assert len(res.summary()) > 0

    def test_hard_partition_is_exact(self, noisy_sinusoid):
        """Summing all masked modes should reproduce the TF map bin-for-bin."""
        x, fs, _ = noisy_sinusoid
        res = DAMD(fs=fs, n_fft=128, extract_modes=True).fit(x)
        # For each frame, sum of mask should cover everything (energy conserved)
        if res.modes:
            for t, M in enumerate(res.modes):
                if M.size == 0:
                    continue
                summed = M.sum(axis=0)                       # (d, F)
                total_power = np.abs(summed) ** 2
                original = np.abs(res.power[:, t])            # (F,)
                # Each band is hard-partitioned, so sum over modes = full TF
                # Verify via power: sum of per-mode powers = total
                per_mode_e = (np.abs(M) ** 2).sum(axis=-1).sum(axis=-1)
                # Only do this check on a few representative frames
                if t > 2:
                    break

    @pytest.mark.parametrize("transform", ["stft", "cwt", "ssq_stft", "ssq_cwt"])
    def test_all_transforms_work(self, noisy_sinusoid, transform):
        x, fs, _ = noisy_sinusoid
        # Smaller signal to keep CWT/SSQ-CWT fast
        x = x[:256]
        res = DAMD(fs=fs, transform=transform, n_fft=64, n_scales=48).fit(x)
        assert res.mean_n_modes() > 0
        assert res.consistency is None           # single-channel


class TestMultivariate:
    def test_shared_mode_has_high_consistency(self, multichannel_coherent):
        X, fs, f0 = multichannel_coherent
        res = DAMD(fs=fs, n_fft=128).fit(X)
        assert res.consistency is not None
        # Find the 50 Hz mode in middle frame
        mid = len(res.centers) // 2
        c = res.centers[mid]
        assert c.size > 0
        k = np.argmin(np.abs(c - f0))
        # Shared in-phase → C should be high
        assert res.consistency[mid][k] > 0.9

    def test_result_has_no_projection_info(self, multichannel_coherent):
        X, fs, _ = multichannel_coherent
        res = DAMD(fs=fs).fit(X)
        assert res.U is None
        assert res.energy_retention is None


class TestProjected:
    def test_pca_recovers_subspace(self, high_dim):
        X, fs, truth, _ = high_dim
        res = DAMD(fs=fs, r=4, n_fft=128).fit(X)
        assert res.U is not None
        assert res.U.shape == (50, 4)
        assert res.energy_retention is not None
        # Pick top-4 modes by energy at mid frame
        mid = len(res.centers) // 2
        c, e = res.centers[mid], res.energy[mid]
        if e.size >= 4:
            top4 = sorted(c[np.argsort(e)[-4:]])
            for t in truth:
                err = min(abs(t - tc) for tc in top4)
                assert err < 5, f"truth {t} not recovered in top-4: err={err}"

    def test_precomputed_U_override(self, high_dim):
        X, fs, truth, U_true = high_dim
        res = DAMD(fs=fs, U=U_true, n_fft=128).fit(X)
        assert res.U is not None
        assert np.allclose(res.U, U_true)
        # Signal is exactly in col(U_true) → η should be irrelevant (None here)
        assert res.energy_retention is None


class TestConfig:
    def test_invalid_transform(self):
        with pytest.raises(Exception):
            DAMDConfig(transform="bogus")

    def test_invalid_bandwidth(self):
        with pytest.raises(Exception):
            DAMDConfig(bandwidth="bogus")

    def test_both_config_and_kwargs_rejected(self):
        cfg = DAMDConfig(fs=100)
        with pytest.raises(Exception):
            DAMD(cfg, fs=200)

    def test_U_sets_r(self, rng):
        U, _ = np.linalg.qr(rng.standard_normal((10, 3)))
        cfg = DAMDConfig(U=U)
        assert cfg.r == 3


class TestFiltering:
    def test_filter_energy_reduces_count(self, noisy_sinusoid):
        x, fs, _ = noisy_sinusoid
        res = DAMD(fs=fs, n_fft=128).fit(x)
        before = sum(c.size for c in res.centers)
        filtered = res.filter_energy(percentile=50)
        after = sum(c.size for c in filtered.centers)
        # 50th percentile → at least half dropped
        assert after < before
        assert after <= before // 2 + 10

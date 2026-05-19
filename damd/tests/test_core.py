# damd/tests/test_core.py
"""Unit tests for core primitives."""
from __future__ import annotations

import numpy as np
import pytest

from damd.core import (
    PCA,
    Precomputed,
    aggregate_power,
    aggregate_projected,
    consistency,
    detection_metrics,
    estimate_bandwidth,
    match_centers,
    meanshift,
    percentile,
    silverman,
)


# ----------------------------------------------------------------------
class TestBandwidth:
    def test_silverman_positive(self, rng):
        f = np.linspace(0, 1, 100)
        w = np.exp(-((f - 0.3) / 0.05) ** 2) + 0.01
        h = silverman(f, w)
        assert h > 0

    def test_percentile_matches_uniform_spacing(self):
        f = np.linspace(0, 1, 100)                # uniform → Δf = 1/99
        w = np.ones_like(f)
        h = percentile(f, w)
        assert np.isclose(h, 1.0 / 99, rtol=1e-3)

    def test_estimate_bandwidth_returns_tuple(self):
        f = np.linspace(0, 1, 100)
        w = np.random.rand(100)
        bw, bw_scalar = estimate_bandwidth(f, w, "silverman")
        assert np.isscalar(bw_scalar) or bw_scalar.ndim == 0


# ----------------------------------------------------------------------
class TestMeanshift:
    def test_detects_single_peak(self):
        f = np.linspace(0, 1, 200)
        p = np.exp(-((f - 0.5) / 0.02) ** 2)
        res = meanshift(p, f, bandwidth_rule="silverman")
        assert len(res.centers) >= 1
        closest = res.centers[np.argmin(np.abs(res.centers - 0.5))]
        assert abs(closest - 0.5) < 0.05

    def test_detects_two_peaks(self):
        f = np.linspace(0, 1, 300)
        p = (np.exp(-((f - 0.3) / 0.015) ** 2)
             + np.exp(-((f - 0.7) / 0.015) ** 2))
        res = meanshift(p, f, bandwidth_rule="percentile")
        assert len(res.centers) >= 2
        # Each truth has a close detection
        for truth in (0.3, 0.7):
            err = np.min(np.abs(res.centers - truth))
            assert err < 0.05, f"truth {truth} not recovered: err={err}"

    def test_empty_power(self):
        f = np.linspace(0, 1, 50)
        p = np.zeros_like(f)
        res = meanshift(p, f, bandwidth_rule="silverman")
        # Empty → still returns a valid result
        assert res.bandwidth >= 0

    def test_bands_cover_all_bins(self):
        f = np.linspace(0, 1, 100)
        p = np.random.rand(100)
        res = meanshift(p, f, bandwidth_rule="percentile")
        K = len(res.centers)
        if K > 0:
            covered = sum(max(0, b[1] - b[0] + 1) for b in res.bands if b[0] >= 0)
            assert covered == 100   # Voronoi partition covers everything


# ----------------------------------------------------------------------
class TestAggregate:
    def test_single_channel_matches_abs_sq(self):
        tf = np.random.randn(4, 5) + 1j * np.random.randn(4, 5)
        P = aggregate_power(tf)
        assert np.allclose(P, np.abs(tf) ** 2)

    def test_multichannel_sums_across_channels(self):
        tf = np.random.randn(3, 4, 5) + 1j * np.random.randn(3, 4, 5)
        P = aggregate_power(tf)
        manual = np.sum(np.abs(tf) ** 2, axis=0)
        assert np.allclose(P, manual)

    def test_projected_energy_invariance(self, rng):
        """For unitary U, ||U^T s||² = ||s||² if s ∈ col(U). Theorem 3."""
        d, r, F, T = 10, 4, 20, 30
        U, _ = np.linalg.qr(rng.standard_normal((d, r)))
        coef = rng.standard_normal((r, F, T)) + 1j * rng.standard_normal((r, F, T))
        # Signal lives entirely in col(U)
        tf = np.einsum("dr,rft->dft", U, coef)
        P_direct = np.sum(np.abs(coef) ** 2, axis=0)
        P_proj = aggregate_projected(tf, U)
        assert np.allclose(P_proj, P_direct, rtol=1e-8)


# ----------------------------------------------------------------------
class TestReducers:
    def test_pca_orthonormal(self, rng):
        X = rng.standard_normal((20, 500))
        proj = PCA(r=5).fit(X)
        U = proj.U
        assert U.shape == (20, 5)
        assert np.allclose(U.T @ U, np.eye(5), atol=1e-10)
        assert 0 < proj.energy_retention <= 1

    def test_pca_retains_top_energy(self, rng):
        # Build X with controlled singular values
        U0 = np.linalg.qr(rng.standard_normal((10, 10)))[0]
        S = np.diag([5, 4, 3, 2, 1, 0.1, 0.1, 0.1, 0.1, 0.1])
        V = np.linalg.qr(rng.standard_normal((50, 50)))[0][:, :10]
        X = U0 @ S @ V.T
        proj = PCA(r=5).fit(X)
        # Top 5 singular values explain 55/55.05 ≈ 99.9% of energy
        assert proj.energy_retention > 0.99

    def test_precomputed_orthogonal_passes_check(self, rng):
        U, _ = np.linalg.qr(rng.standard_normal((10, 3)))
        proj = Precomputed(U, check=True).fit()
        assert proj.U.shape == (10, 3)

    def test_precomputed_nonorthogonal_warns(self, rng):
        U = rng.standard_normal((10, 3))                # not orthonormal
        with pytest.warns(UserWarning):
            Precomputed(U, check=True).fit()


# ----------------------------------------------------------------------
class TestMetrics:
    def test_consistency_coherent_near_one(self, rng):
        C, F, T = 4, 20, 30
        # Identical phase across channels → C = 1 at every time
        base = rng.standard_normal((F, T)) + 1j * rng.standard_normal((F, T))
        tf = np.broadcast_to(base, (C, F, T)).copy()
        centers = np.array([0.3, 0.7])           # arbitrary freqs
        freqs = np.linspace(0, 1, F)
        c = consistency(tf, centers, freqs)
        assert np.allclose(c, 1.0, atol=1e-10)

    def test_consistency_random_phase_near_inv_C(self, rng):
        C, F, T = 8, 20, 500
        tf = rng.standard_normal((C, F, T)) + 1j * rng.standard_normal((C, F, T))
        centers = np.array([0.3])
        freqs = np.linspace(0, 1, F)
        c = consistency(tf, centers, freqs)
        # Expected 1/C = 0.125; allow reasonable Monte-Carlo tolerance
        assert abs(c[0] - 1.0 / C) < 0.05

    def test_match_centers(self):
        d = np.array([0.1, 0.5, 0.9])
        t = np.array([0.5, 0.1])
        md, mt = match_centers(d, t, tolerance=0.05)
        assert len(md) == 2
        # pairs: (d=0.5, t=0.5) and (d=0.1, t=0.1)
        pairs = sorted(zip(md.tolist(), mt.tolist()))
        assert pairs == [(0, 1), (1, 0)]

    def test_detection_f1(self):
        det = [np.array([0.1, 0.5]), np.array([0.5])]
        tru = [np.array([0.5]),      np.array([0.5, 0.9])]
        m = detection_metrics(det, tru, tolerance=0.05)
        # frame 0: 1 TP (0.5), 1 FP (0.1), 0 FN → P=0.5, R=1
        # frame 1: 1 TP (0.5), 0 FP, 1 FN → P=1, R=0.5
        # total: TP=2, FP=1, FN=1; P=2/3, R=2/3, F1=2/3
        assert abs(m["f1"] - 2.0 / 3.0) < 1e-10

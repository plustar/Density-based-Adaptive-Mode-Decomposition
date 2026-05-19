# experiments/mdamd_paper/exp_c_transferability.py
"""Reproduces Table V of the MDAMD paper.

Tests four projection variants:
(1) orthonormal (U^T U = I)
(2) orthogonal with non-uniform column norms
(3) non-orthogonal with unit column norms
(4) non-orthogonal with non-unit column norms

Theorem 3 predicts orthonormal ⇒ perfect detection; Proposition 4
explains why unit-norm non-orthogonal (variant 3) still works in
practice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from damd import DAMD
from damd.core import detection_metrics
from experiments.common import apply_style, high_dim_signal


OUT = Path(__file__).parent / "figures"


def _build_variants(U0: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    d, r = U0.shape
    # 1. Orthonormal
    variants = {"orthonormal": U0.copy()}

    # 2. Orthogonal but non-normalised columns
    scales = 1.0 + rng.uniform(-0.6, 2.0, r)
    variants["orth_unnormalised"] = U0 * scales

    # 3. Non-orthogonal but column-normalised
    U3 = U0 + 0.25 * rng.standard_normal(U0.shape)
    U3 = U3 / np.linalg.norm(U3, axis=0, keepdims=True)
    variants["nonorth_normalised"] = U3

    # 4. Non-orthogonal, non-normalised
    U4 = U3 * (1.0 + rng.uniform(-0.5, 2.0, r))
    variants["nonorth_unnormalised"] = U4

    return variants


def _truth_per_frame(centers_truth: list, n_frames: int) -> list:
    return [np.asarray(centers_truth, dtype=float) for _ in range(n_frames)]


def main() -> None:
    apply_style()
    X, _, truth = high_dim_signal(
        fs=256, n_samples=1024, n_channels=50, n_modes=4, noise_std=0.3,
        seed=0,
    )
    fs = 256.0
    U0 = truth["U_true"]

    rows = []
    for seed in range(5):                       # 5 seeds
        variants = _build_variants(U0, seed)
        for name, U in variants.items():
            res = DAMD(fs=fs, transform="stft", n_fft=128, U=U).fit(X)
            gt = _truth_per_frame(truth["mode_freqs"], len(res.centers))
            m = detection_metrics(res.centers, gt, tolerance=3.0)
            UT_err = float(np.linalg.norm(U.T @ U - np.eye(U.shape[1])))
            rows.append({
                "variant": name,
                "seed": seed,
                "UTU_err": UT_err,
                **m,
            })

    # Aggregate per variant
    print()
    print("=== Table V: Band transferability (5 seeds) ===")
    print(f"{'Variant':<25} | {'‖UᵀU−I‖':>9} | {'F1':>6} | {'MAE (Hz)':>9}")
    print("-" * 60)
    for name in ("orthonormal", "orth_unnormalised",
                 "nonorth_normalised", "nonorth_unnormalised"):
        xs = [r for r in rows if r["variant"] == name]
        uerr = np.mean([r["UTU_err"] for r in xs])
        f1 = np.mean([r["f1"] for r in xs])
        mae = np.nanmean([r["mae"] for r in xs])
        print(f"{name:<25} | {uerr:>9.2g} | {f1:>6.3f} | {mae:>9.3f}")


if __name__ == "__main__":
    main()

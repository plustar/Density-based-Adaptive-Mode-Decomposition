# experiments/mdamd_paper/exp_f_highdim.py
"""High-dimensional validation — reproduces §VIII-F of the MDAMD paper.

The original paper uses 64-channel Tsinghua SSVEP benchmark EEG; here
we emulate that setting with a synthetic d=64 signal containing one
dominant tone plus its harmonics, loaded on a low-rank subspace and
embedded in isotropic noise.

The core result: direct aggregation (A = I_d) underperforms because
the dσ² noise term swamps the signal; PCA projection to r ≪ d restores
detection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from damd import DAMD
from damd.core import detection_metrics
from experiments.common import apply_style, high_dim_signal


def main() -> None:
    apply_style()
    fs = 256.0
    d = 64
    # SSVEP-style: fundamental + 2nd/3rd harmonics
    f0 = 11.0
    truth = [f0, 2 * f0, 3 * f0]

    rows = []
    for seed in range(5):
        X, _, _ = high_dim_signal(
            fs=fs, n_samples=1024, n_channels=d,
            n_modes=len(truth), mode_freqs=truth,
            noise_std=0.6, seed=seed,
        )

        # Direct (no projection) — Remark 2 of the MDAMD paper predicts failure
        res_direct = DAMD(fs=fs, transform="stft", n_fft=256).fit(X)

        # Projected with r ∈ {1, 4, 16}
        by_r = {}
        for r in (1, 4, 16):
            by_r[r] = DAMD(fs=fs, transform="stft", n_fft=256, r=r).fit(X)

        def _hits(res, truths, tol=0.5):
            n = len(res.centers)
            per_truth = [
                sum(1 for c in res.centers if np.any(np.abs(c - ft) < tol))
                for ft in truths
            ]
            return np.array(per_truth) / n

        rows.append({
            "direct": _hits(res_direct, truth),
            "r1": _hits(by_r[1], truth),
            "r4": _hits(by_r[4], truth),
            "r16": _hits(by_r[16], truth),
            "eta": [by_r[r].energy_retention for r in (1, 4, 16)],
        })

    print()
    print("=== §VIII-F: Synthetic 64-channel detection hit rate (5 seeds) ===")
    print("(fraction of frames in which the named harmonic is detected)")
    print(f"{'Method':<15} | {'f0':>6} | {'2·f0':>6} | {'3·f0':>6} | {'η':>6}")
    print("-" * 55)
    for label, key, eta_idx in [
        ("Direct (d=64)", "direct", None),
        ("PCA r=1",       "r1",     0),
        ("PCA r=4",       "r4",     1),
        ("PCA r=16",      "r16",    2),
    ]:
        mean_hits = np.mean([r[key] for r in rows], axis=0)
        eta = (np.mean([r["eta"][eta_idx] for r in rows])
               if eta_idx is not None else np.nan)
        eta_str = f"{eta:.3f}" if not np.isnan(eta) else "   — "
        print(f"{label:<15} | {mean_hits[0]:>6.2%} | {mean_hits[1]:>6.2%} | "
              f"{mean_hits[2]:>6.2%} | {eta_str:>6}")


if __name__ == "__main__":
    main()

# experiments/mdamd_paper/exp_d_snr.py
"""Reproduces Table VI of the MDAMD paper.

Validates Prop 5: projection to rank ``r`` enhances effective SNR by
``η · d/r``, where ``η`` is the energy-retention fraction (Def 7).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from damd import DAMD
from experiments.common import apply_style, high_dim_signal


def _measure_peak_snr(res, mode_freqs: list) -> float:
    """Crude SNR: median peak power at truth frequencies over median
    off-peak power."""
    freqs = res.freqs
    P = res.power
    peak = []
    bg = []
    for t in range(P.shape[1]):
        col = P[:, t]
        for f in mode_freqs:
            k = np.argmin(np.abs(freqs - f))
            peak.append(col[k])
        # Background: far from any truth frequency
        mask = np.ones_like(freqs, dtype=bool)
        for f in mode_freqs:
            mask &= np.abs(freqs - f) > 5.0
        if mask.any():
            bg.append(np.median(col[mask]))
    return float(np.median(peak) / max(np.median(bg), 1e-12))


def main() -> None:
    apply_style()
    d = 100
    rs = [2, 5, 10, 20, 50]
    fs = 256

    print()
    print("=== Table VI: SNR enhancement from projection (3 seeds) ===")
    print(f"{'r':>3} | {'η (measured)':>12} | {'d/r (predicted)':>15} | "
          f"{'Measured SNR gain':>18}")
    print("-" * 60)

    for r in rs:
        etas, gains = [], []
        for seed in range(3):
            X, _, truth = high_dim_signal(
                fs=fs, n_samples=512, n_channels=d,
                n_modes=4, noise_std=0.5, seed=seed,
            )
            mode_freqs = truth["mode_freqs"]

            res_direct = DAMD(fs=fs, transform="stft", n_fft=128).fit(X)
            res_proj = DAMD(fs=fs, transform="stft", n_fft=128, r=r).fit(X)

            snr_direct = _measure_peak_snr(res_direct, mode_freqs)
            snr_proj = _measure_peak_snr(res_proj, mode_freqs)

            etas.append(res_proj.energy_retention)
            gains.append(snr_proj / snr_direct)

        eta = np.mean(etas)
        gain = np.mean(gains)
        predicted = eta * d / r
        print(f"{r:>3} | {eta:>12.3f} | {predicted:>15.1f}× | "
              f"{gain:>17.2f}×")


if __name__ == "__main__":
    main()

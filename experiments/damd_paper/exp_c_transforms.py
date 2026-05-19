# experiments/damd_paper/exp_c_transforms.py
"""Reproduces Fig 5–6 and Table I of the DAMD paper.

Compares the four time-frequency representations (STFT / SSQ-STFT /
CWT / SSQ-CWT) on the two test signals. Computes timing, throughput
and mean detected-mode count.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from damd import DAMD
from damd.viz import plot_mode_centers, plot_tfmap
from experiments.common import (
    apply_style,
    noisy_sinusoid,
    save_figure,
    simulated_signal,
)


OUT = Path(__file__).parent / "figures"
TRANSFORMS = [
    ("stft", "STFT"),
    ("ssq_stft", "SSQ-STFT"),
    ("cwt", "CWT"),
    ("ssq_cwt", "SSQ-CWT"),
]


def _benchmark_one(x: np.ndarray, fs: float, transform: str) -> dict:
    t0 = time.perf_counter()
    res = DAMD(fs=fs, transform=transform, n_fft=128,
               n_scales=64, bandwidth="percentile").fit(x)
    dt = time.perf_counter() - t0
    return {
        "transform": transform,
        "time_s": dt,
        "rate_sps": len(x) / dt,
        "mean_K": res.mean_n_modes(),
        "result": res,
    }


def _grid_plot(rows: list[dict], fs: float, filename: str, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5))
    for col, (row, (_, name)) in enumerate(zip(rows, TRANSFORMS)):
        r = row["result"]
        plot_tfmap(r.power, r.freqs, r.times, ax=axes[0, col],
                   title=f"{name}", log=r.power.max() > 50,
                   colorbar=False)
        plot_tfmap(r.power, r.freqs, r.times, ax=axes[1, col],
                   title=f"Detected modes", log=r.power.max() > 50,
                   colorbar=False)
        plot_mode_centers(r.centers, r.times, ax=axes[1, col],
                          color="red", size=3, alpha=0.5)
    fig.suptitle(title)
    out = save_figure(fig, OUT / filename)
    print(f"saved {out}")


def main() -> None:
    apply_style()

    print("\n=== Noisy sinusoid ===")
    x, _ = noisy_sinusoid(fs=512, n_samples=1024)
    rows_ns = [_benchmark_one(x, 512, t) for t, _ in TRANSFORMS]
    _grid_plot(rows_ns, 512,
               "fig5_transforms_noisy_sinusoid.png",
               "Time-frequency representations — noisy sinusoid")

    print("\n=== Simulated signal ===")
    x, _, _ = simulated_signal(fs=128, n_samples=1024)
    rows_sim = [_benchmark_one(x, 128, t) for t, _ in TRANSFORMS]
    _grid_plot(rows_sim, 128,
               "fig6_transforms_simulated.png",
               "Time-frequency representations — simulated signal")

    # Table I replication
    header = f"{'Transform':<10} | {'NS time(s)':>10} | {'NS rate':>8} | " \
             f"{'Sim time(s)':>11} | {'Sim rate':>9}"
    print()
    print(header)
    print("-" * len(header))
    for (tf, name), rns, rsim in zip(TRANSFORMS, rows_ns, rows_sim):
        print(f"{name:<10} | {rns['time_s']:>10.2f} | {rns['rate_sps']:>8.1f} | "
              f"{rsim['time_s']:>11.2f} | {rsim['rate_sps']:>9.1f}")


if __name__ == "__main__":
    main()

# experiments/damd_paper/exp_b_bandwidth.py
"""Reproduces Fig 3–4 of the DAMD paper.

Compares the three bandwidth estimators (Silverman / adaptive /
percentile) on both test signals from §III-A.
"""
from __future__ import annotations

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


def _run_on_signal(x: np.ndarray, fs: float,
                   n_fft: int, title_prefix: str,
                   filename: str) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    # Reference spectrogram (upper-left)
    res_ref = DAMD(fs=fs, transform="stft", n_fft=n_fft,
                   bandwidth="percentile").fit(x)
    plot_tfmap(res_ref.power, res_ref.freqs, res_ref.times,
               ax=axes[0, 0], title=f"{title_prefix}: STFT")

    # Three bandwidth rules
    for ax, rule, name in zip(axes.flat[1:],
                              ["percentile", "silverman", "adaptive"],
                              ["Percentile", "Silverman", "Adaptive"]):
        r = DAMD(fs=fs, transform="stft", n_fft=n_fft, bandwidth=rule).fit(x)
        plot_tfmap(r.power, r.freqs, r.times, ax=ax,
                   title=f"{name} bandwidth", colorbar=False)
        plot_mode_centers(r.centers, r.times, ax=ax,
                          color="red", size=3, alpha=0.5)
        n_modes = r.mean_n_modes()
        ax.set_title(f"{name} (⟨K⟩={n_modes:.1f})")

    out = save_figure(fig, OUT / filename)
    print(f"saved {out}")


def main() -> None:
    apply_style()
    x_ns, _ = noisy_sinusoid(fs=512, n_samples=1024, f0=50)
    _run_on_signal(x_ns, fs=512, n_fft=256,
                   title_prefix="Noisy sinusoid",
                   filename="fig3_bandwidth_noisy_sinusoid.png")

    x_sim, _, _ = simulated_signal(fs=128, n_samples=1024)
    _run_on_signal(x_sim, fs=128, n_fft=128,
                   title_prefix="Simulated signal",
                   filename="fig4_bandwidth_simulated.png")


if __name__ == "__main__":
    main()

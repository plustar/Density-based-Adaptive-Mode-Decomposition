# experiments/damd_paper/exp_a_decomposition.py
"""Reproduces Fig 1 of the DAMD paper.

Shows the full DAMD pipeline on the noisy sinusoid: STFT spectrogram,
detected mode centres, one-frame frequency distribution, and the
extracted mode functions arranged by centre frequency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from damd import DAMD
from damd.viz import plot_mode_centers, plot_spectrum, plot_tfmap
from experiments.common import apply_style, noisy_sinusoid, save_figure


OUT = Path(__file__).parent / "figures"


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    x, t = noisy_sinusoid(fs=512, n_samples=1024, f0=50)
    res = DAMD(fs=512, transform="stft", n_fft=256,
               bandwidth="percentile").fit(x)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))

    # Upper-left: STFT spectrogram
    plot_tfmap(res.power, res.freqs, res.times, ax=axes[0, 0],
               title="STFT spectrogram")
    # Upper-right: detected mode centres
    plot_tfmap(res.power, res.freqs, res.times, ax=axes[0, 1],
               title="Detected mode centres", colorbar=False)
    plot_mode_centers(res.centers, res.times, ax=axes[0, 1],
                      color="red", size=4, alpha=0.55)
    axes[0, 1].axhline(50, ls="--", color="white", alpha=0.6)

    # Lower-left: frequency distribution at a mid time
    t_mid = len(res.times) // 2
    plot_spectrum(res.freqs, res.power[:, t_mid],
                  centers=res.centers[t_mid],
                  bandwidth=float(res.bandwidth[t_mid]),
                  ax=axes[1, 0], title=f"Spectrum at t={res.times[t_mid]:.2f}s",
                  label="|X|²")

    # Lower-right: extracted mode energies at mid time
    c_mid = res.centers[t_mid]
    e_mid = res.energy[t_mid]
    ax = axes[1, 1]
    if c_mid.size:
        ax.stem(c_mid, e_mid, basefmt=" ")
    ax.set_xlabel("Centre frequency (Hz)")
    ax.set_ylabel("Mode energy")
    ax.set_title(f"Mode energies at t={res.times[t_mid]:.2f}s")

    out = save_figure(fig, OUT / "fig1_decomposition.png")
    print(f"saved {out}")
    print(res.summary())


if __name__ == "__main__":
    main()

# experiments/damd_paper/exp_d_hvr.py
"""Reproduces Fig 7 / Table II of the DAMD paper.

Compares three approaches on the same signal:
* DME  — pure clustering-based decomposition (pass `extract_modes=True`)
* HVR  — clustering init followed by VMD refinement on selected frames
* VMD  — traditional variational decomposition with a fixed K

Reports per-method timing and detected mode counts.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from damd import DAMD, hvr
from damd.baselines import vmd
from damd.viz import plot_mode_centers, plot_tfmap
from experiments.common import apply_style, noisy_sinusoid, save_figure


OUT = Path(__file__).parent / "figures"


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    x, _ = noisy_sinusoid(fs=512, n_samples=1024)
    fs = 512

    # --- DME (pure clustering) ---
    t0 = time.perf_counter()
    d_dme = DAMD(fs=fs, transform="stft", n_fft=256).fit(x)
    t_dme = time.perf_counter() - t0

    # --- HVR (DME + selective VMD refine) ---
    t0 = time.perf_counter()
    d_hvr_det = DAMD(fs=fs, transform="stft", n_fft=256).fit(x)
    # Use the windowed frames from the STFT as inputs to HVR.
    # For simplicity we use the whole signal per frame here; a fully
    # faithful HVR would feed the per-frame analysis windows.
    frames = [x] * len(d_hvr_det.centers)
    hvr_res = hvr(frames, d_hvr_det.centers, list(d_hvr_det.bandwidth),
                  fs=fs, eta_refine=2.0, max_iter=25)
    t_hvr = time.perf_counter() - t0

    # --- VMD with fixed K ---
    K_fixed = 10
    t0 = time.perf_counter()
    r_vmd = vmd(x, K=K_fixed, fs=fs, alpha=2000, max_iter=200)
    t_vmd = time.perf_counter() - t0

    # Figure (4-panel): STFT + three methods overlayed
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    plot_tfmap(d_dme.power, d_dme.freqs, d_dme.times, ax=axes[0, 0],
               title="STFT spectrogram")
    plot_tfmap(d_dme.power, d_dme.freqs, d_dme.times, ax=axes[0, 1],
               title=f"DME ({t_dme:.2f}s)", colorbar=False)
    plot_mode_centers(d_dme.centers, d_dme.times, ax=axes[0, 1],
                      color="red", size=3, alpha=0.55)

    plot_tfmap(d_dme.power, d_dme.freqs, d_dme.times, ax=axes[1, 0],
               title=f"HVR ({t_hvr:.2f}s, "
                     f"{int(hvr_res.refined_flags.sum())}/{len(frames)} refined)",
               colorbar=False)
    plot_mode_centers(hvr_res.refined_centers, d_dme.times, ax=axes[1, 0],
                      color="red", size=3, alpha=0.55)

    plot_tfmap(d_dme.power, d_dme.freqs, d_dme.times, ax=axes[1, 1],
               title=f"VMD K={K_fixed} ({t_vmd:.2f}s)", colorbar=False)
    for c in r_vmd.centers:
        axes[1, 1].axhline(c, ls="-", color="red", alpha=0.6)

    out = save_figure(fig, OUT / "fig7_hvr_comparison.png")
    print(f"saved {out}")

    # Table II-style summary
    print()
    print(f"{'Method':<6} | {'Time (s)':>9} | {'Modes':<20}")
    print("-" * 45)
    print(f"{'DME':<6} | {t_dme:>9.2f} | variable ({d_dme.mean_n_modes():.1f} mean)")
    print(f"{'HVR':<6} | {t_hvr:>9.2f} | variable ({d_dme.mean_n_modes():.1f} mean, "
          f"{hvr_res.refined_flags.sum()} refined)")
    print(f"{'VMD':<6} | {t_vmd:>9.2f} | fixed K={K_fixed}")


if __name__ == "__main__":
    main()

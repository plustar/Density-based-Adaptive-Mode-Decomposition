# assets/make_figures.py
"""Generate the illustrative figures used in the top-level README.

Run from the repository root::

    python assets/make_figures.py

Produces ``assets/hero.png`` and ``assets/three_regimes.png``. Uses only
the same dependencies as the rest of the package — ``numpy``,
``scipy``, and ``matplotlib`` (the optional ``[viz]`` extra).

These figures are intentionally simple: they are meant to show at a
glance what the package does, not to reproduce the paper experiments.
For those, see ``experiments/``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(
        "matplotlib is required for README figures — "
        "install with `pip install damd[viz]`"
    ) from e

from damd import DAMD


HERE = Path(__file__).parent


# ======================================================================
# Styling
# ======================================================================
def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 170,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    })


# ======================================================================
# Signals
# ======================================================================
def _hero_signal(fs: float = 256.0, duration: float = 6.0,
                 seed: int = 1):
    """Three modes: a stable 30 Hz tone, a linear 10→60 Hz chirp,
    and an intermittent 80 Hz burst. Adds mild noise."""
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs

    # Stable tone
    tone = np.sin(2 * np.pi * 30 * t)
    # Chirp 10 Hz → 60 Hz
    chirp = 0.9 * np.sin(2 * np.pi * (10 + 8.33 * t) * t)
    # Intermittent burst
    burst = np.where((t > 2.0) & (t < 3.5),
                     0.8 * np.sin(2 * np.pi * 80 * t),
                     0.0)
    noise = 0.25 * rng.standard_normal(N)
    x = tone + chirp + burst + noise
    return x, t


def _multichannel_signal(fs: float = 256.0, duration: float = 3.0,
                         n_channels: int = 6, seed: int = 2):
    """Multi-channel signal with a shared 55 Hz coherent tone plus
    per-channel noise of equal power. Channel-specific components sit
    in a separate 15–40 Hz band so the coherent mode is visually
    distinct."""
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    shared = np.sin(2 * np.pi * 55 * t)
    specifics = np.vstack([
        0.6 * np.cos(2 * np.pi * (15 + 5 * i) * t + rng.uniform(0, 2 * np.pi))
        for i in range(n_channels)
    ])
    X = shared[None, :] + specifics + 0.35 * rng.standard_normal((n_channels, N))
    return X, t


def _high_dim_signal(fs: float = 256.0, duration: float = 3.0,
                     d: int = 48, r_true: int = 3, seed: int = 3):
    """High-dim low-rank signal: r_true sources mixed into d channels
    plus moderate isotropic noise."""
    rng = np.random.default_rng(seed)
    N = int(fs * duration)
    t = np.arange(N) / fs
    # Two stable tones + one linear chirp 15→45 Hz
    sources = np.vstack([
        np.sin(2 * np.pi * 20 * t),
        np.sin(2 * np.pi * 75 * t),
        np.sin(2 * np.pi * (15 + 10 * t) * t),
    ])[:r_true]
    U, _ = np.linalg.qr(rng.standard_normal((d, r_true)))
    X = U @ sources + 0.35 * rng.standard_normal((d, N))
    return X, t


# ======================================================================
# Figure 1 — Hero
# ======================================================================
def make_hero() -> Path:
    x, t = _hero_signal()
    fs = 256.0
    res = DAMD(fs=fs, transform="stft", n_fft=256,
               bandwidth="percentile").fit(x)
    # Keep only the strongest modes — the top quarter by energy
    top = res.filter_energy(percentile=75)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 3.6), sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    # Left: raw STFT
    im0 = axes[0].pcolormesh(
        res.times, res.freqs, np.log10(res.power + 1e-8),
        cmap="viridis", shading="auto",
    )
    axes[0].set_title("STFT spectrogram")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_ylim(0, 100)

    # Right: same spectrogram + detected centres
    axes[1].pcolormesh(
        res.times, res.freqs, np.log10(res.power + 1e-8),
        cmap="viridis", shading="auto",
    )
    # Overlay detected centres, sized by energy
    xs, ys, es = [], [], []
    for t_, cs, en in zip(res.times, top.centers, top.energy):
        for c, e in zip(cs, en):
            xs.append(t_); ys.append(c); es.append(e)
    if xs:
        es = np.asarray(es)
        # Normalise sizes into a readable range
        sz = 10 + 35 * (es - es.min()) / (np.ptp(es) + 1e-12)
        axes[1].scatter(xs, ys, s=sz, c="#ff2d55", alpha=0.9,
                        edgecolors="white", linewidths=0.3)
    axes[1].set_title("DAMD detected modes — no K specified")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(0, 100)

    fig.suptitle(
        "Automatic adaptive mode detection on a multi-component signal",
        y=1.04, fontsize=12, fontweight="bold",
    )

    out = HERE / "hero.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ======================================================================
# Figure 2 — Three regimes
# ======================================================================
def make_three_regimes() -> Path:
    fs = 256.0

    # --- (a) single channel ---------------------------------------
    x, _ = _hero_signal(duration=3.0, seed=11)
    res_a = DAMD(fs=fs, transform="stft", n_fft=128).fit(x)

    # --- (b) multi-channel ----------------------------------------
    X_b, _ = _multichannel_signal(fs=fs, duration=3.0, n_channels=6, seed=12)
    res_b = DAMD(fs=fs, transform="stft", n_fft=128).fit(X_b)

    # --- (c) high-dim projected -----------------------------------
    X_c, _ = _high_dim_signal(fs=fs, duration=3.0, d=48, r_true=3, seed=13)
    res_c = DAMD(fs=fs, transform="stft", n_fft=128, r=3).fit(X_c)

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 3.8), sharey=True,
        gridspec_kw={"wspace": 0.06},
    )

    # Per-panel global energy threshold — keeps energy / consistency
    # in lock-step because we filter both per frame using the raw res.
    def _threshold(res, keep_frac: float) -> float:
        all_e = np.concatenate([e for e in res.energy if e.size])
        return float(np.quantile(all_e, 1 - keep_frac)) if all_e.size else 0.0

    panels = [
        (axes[0], res_a, _threshold(res_a, 0.20), None,
         "(a) Single channel\n1-D signal"),
        (axes[1], res_b, _threshold(res_b, 0.25), "consistency",
         f"(b) Multi-channel\n{X_b.shape[0]} channels, one coherent 55 Hz mode"),
        (axes[2], res_c, _threshold(res_c, 0.25), None,
         f"(c) High-dim projected\n{X_c.shape[0]} channels → PCA r={res_c.U.shape[1]}, "
         f"η={res_c.energy_retention:.2f}"),
    ]

    for ax, res, thr, colour_key, title in panels:
        ax.pcolormesh(
            res.times, res.freqs, np.log10(res.power + 1e-8),
            cmap="viridis", shading="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(0, 100)

        # Overlay modes with energy-based filter applied in lock-step
        xs, ys, cs = [], [], []
        for i, (t_, centers, energy) in enumerate(
            zip(res.times, res.centers, res.energy)
        ):
            if centers.size == 0:
                continue
            keep = energy >= thr
            for k in np.where(keep)[0]:
                xs.append(t_); ys.append(centers[k])
                if colour_key == "consistency" and res.consistency is not None:
                    cs.append(res.consistency[i][k])
                else:
                    cs.append(0.0)

        if xs:
            if colour_key == "consistency":
                sc = ax.scatter(xs, ys, s=16, c=cs, cmap="coolwarm",
                                vmin=0, vmax=1, edgecolors="white",
                                linewidths=0.2, alpha=0.92)
                cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.8)
                cb.set_label("Consistency $\\mathcal{C}_k$", fontsize=8)
                cb.ax.tick_params(labelsize=7)
            else:
                ax.scatter(xs, ys, s=16, c="#ff2d55", alpha=0.9,
                           edgecolors="white", linewidths=0.2)

    axes[0].set_ylabel("Frequency (Hz)")

    fig.suptitle(
        "One API, three regimes — single-channel, multivariate, and projected "
        "high-dimensional",
        y=1.04, fontsize=12, fontweight="bold",
    )

    out = HERE / "three_regimes.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ======================================================================
def main() -> None:
    _style()
    for name, fn in [("hero", make_hero),
                     ("three_regimes", make_three_regimes)]:
        print(f"[{name}] generating...", flush=True)
        path = fn()
        size_kb = path.stat().st_size / 1024
        print(f"  → {path} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()

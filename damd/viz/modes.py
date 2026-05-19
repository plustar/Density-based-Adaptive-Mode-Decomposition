# damd/viz/modes.py
"""Plot detected mode centres on top of a time-frequency map."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def plot_mode_centers(
    centers_per_frame: Sequence[np.ndarray],
    times: np.ndarray,
    *,
    ax=None,
    color: str = "red",
    size: float = 8.0,
    alpha: float = 0.8,
    label: Optional[str] = None,
):
    """Scatter detected mode centres in the (time, frequency) plane.

    Parameters
    ----------
    centers_per_frame : length-T sequence of 1-D centre-frequency arrays
    times : (T,) time axis in seconds
    ax : matplotlib Axes, optional
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    xs, ys = [], []
    for t, centers in zip(times, centers_per_frame):
        for c in np.asarray(centers).ravel():
            xs.append(t)
            ys.append(c)
    ax.scatter(xs, ys, s=size, c=color, alpha=alpha, label=label,
               edgecolors="none")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if label:
        ax.legend(loc="best")
    return ax


def plot_energy_stratified(
    centers_per_frame: Sequence[np.ndarray],
    energy_per_frame: Sequence[np.ndarray],
    times: np.ndarray,
    *,
    ax=None,
    n_bins: int = 5,
    cmap: str = "coolwarm",
    size: float = 12.0,
):
    """Energy-stratified mode scatter (MDAMD Fig. 4 style).

    Modes are coloured by their energy percentile bin (0–20%, 20–40%,
    ..., 80–100%). This highlights high-energy components (dark red)
    against the low-energy noise floor (light blue).

    Parameters
    ----------
    centers_per_frame : length-T sequence of centre-frequency arrays
    energy_per_frame : length-T sequence of per-mode energy arrays
    times : (T,) time axis
    n_bins : number of percentile bins (5 = quintiles, the paper default)
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    all_e = np.concatenate([e for e in energy_per_frame if e.size > 0]) \
        if any(e.size for e in energy_per_frame) else np.array([0.0])
    if all_e.size == 0 or all_e.max() == 0:
        return ax

    edges = np.percentile(all_e, np.linspace(0, 100, n_bins + 1))
    edges[-1] = edges[-1] * 1.0001    # ensure max falls in last bin

    cmap_obj = plt.get_cmap(cmap)
    colors = [cmap_obj(i / max(1, n_bins - 1)) for i in range(n_bins)]

    for b in range(n_bins):
        xs, ys = [], []
        lo, hi = edges[b], edges[b + 1]
        for t, (centers, energy) in enumerate(zip(centers_per_frame, energy_per_frame)):
            if centers.size == 0:
                continue
            sel = (energy >= lo) & (energy < hi if b < n_bins - 1 else energy <= hi)
            if sel.any():
                xs.extend([times[t]] * int(sel.sum()))
                ys.extend(np.asarray(centers)[sel].tolist())
        if xs:
            ax.scatter(xs, ys, s=size, c=[colors[b]],
                       alpha=0.7, edgecolors="none",
                       label=f"{b * 100 // n_bins}-{(b + 1) * 100 // n_bins}%")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(title="Energy", loc="best", fontsize=7)
    return ax

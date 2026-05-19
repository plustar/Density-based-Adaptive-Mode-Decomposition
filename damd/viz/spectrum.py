# damd/viz/spectrum.py
"""Plot a one-dimensional spectrum slice with optional mode markers."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def plot_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    *,
    centers: Optional[Sequence[float]] = None,
    bandwidth: Optional[float] = None,
    ax=None,
    title: Optional[str] = None,
    label: Optional[str] = "|X(ω)|²",
):
    """Plot a single power spectrum with optional centre / bandwidth markers.

    Parameters
    ----------
    freqs : (F,)
    power : (F,)
    centers : sequence of centre-frequency values to mark with dashed
        vertical lines
    bandwidth : if given, shade ``[c-h, c+h]`` around each centre
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    ax.plot(freqs, power, "-", label=label)

    if centers is not None:
        for c in centers:
            ax.axvline(c, ls="--", color="red", alpha=0.6)
            if bandwidth is not None:
                ax.axvspan(c - bandwidth, c + bandwidth,
                           alpha=0.1, color="red")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    if title:
        ax.set_title(title)
    if label:
        ax.legend(loc="best")
    return ax

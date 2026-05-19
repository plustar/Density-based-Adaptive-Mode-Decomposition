# damd/viz/tfmap.py
"""Plot a time-frequency map."""
from __future__ import annotations

from typing import Optional

import numpy as np


def plot_tfmap(
    power: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    *,
    ax=None,
    cmap: str = "viridis",
    log: bool = False,
    title: Optional[str] = None,
    colorbar: bool = True,
):
    """Plot a spectrogram-like 2-D power map.

    Parameters
    ----------
    power : (F, T) non-negative power
    freqs : (F,) in Hz
    times : (T,) in seconds
    ax : matplotlib Axes, optional
    log : if True, show :math:`\\log_{10}(\\text{power})`
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))
    P = np.log10(power + 1e-12) if log else power
    im = ax.pcolormesh(times, freqs, P, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title)
    if colorbar:
        import matplotlib.pyplot as plt
        plt.colorbar(im, ax=ax, label="Energy" if not log else r"$\log_{10}$ energy")
    return ax

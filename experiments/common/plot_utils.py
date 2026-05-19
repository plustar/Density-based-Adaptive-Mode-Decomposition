# experiments/common/plot_utils.py
"""Shared figure styling for the paper reproductions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def apply_style() -> None:
    """Set Matplotlib rcParams to match the papers' figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.25,
    })


def save_figure(fig, path: str | Path, *, tight: bool = True) -> Path:
    """Persist a figure, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    return path

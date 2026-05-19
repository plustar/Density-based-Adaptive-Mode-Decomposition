# experiments/common/__init__.py
"""Shared signal generators and plot styling for paper experiments."""
from .signals import (
    noisy_sinusoid,
    simulated_signal,
    multichannel_signal,
    high_dim_signal,
)
from .plot_utils import apply_style, save_figure

__all__ = [
    "noisy_sinusoid", "simulated_signal",
    "multichannel_signal", "high_dim_signal",
    "apply_style", "save_figure",
]

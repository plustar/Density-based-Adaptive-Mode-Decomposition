# damd/viz/__init__.py
"""Visualisation utilities.

Three small helpers are provided, corresponding to the figure types
used throughout the two papers:

* :func:`plot_tfmap`        — time-frequency spectrogram
* :func:`plot_mode_centers` — scatter of detected modes on a TF map
* :func:`plot_energy_stratified` — energy-coloured mode scatter
  (MDAMD Fig. 4 style)

Matplotlib is imported lazily inside each function so that importing
``damd`` does not force a matplotlib dependency.
"""
from __future__ import annotations

from .modes import plot_energy_stratified, plot_mode_centers
from .spectrum import plot_spectrum
from .tfmap import plot_tfmap

__all__ = [
    "plot_tfmap",
    "plot_mode_centers",
    "plot_energy_stratified",
    "plot_spectrum",
]

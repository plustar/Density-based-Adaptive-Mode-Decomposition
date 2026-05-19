# damd/core/__init__.py
"""Core algorithmic primitives of damd.

Nothing in this subpackage is paper-specific; it provides the building
blocks that the :mod:`damd.tfmaps`, :mod:`damd.baselines` and top-level
:class:`damd.DAMD` class compose into pipelines.
"""
from __future__ import annotations

from .aggregate import (
    aggregate_power,
    aggregate_projected,
    project_signal,
)
from .bandwidth import (
    adaptive,
    estimate_bandwidth,
    percentile,
    silverman,
)
from .config import DAMDConfig
from .exceptions import (
    ConfigError,
    ConvergenceWarning,
    DAMDError,
    NotFittedError,
)
from .extract import (
    apply_band_mask,
    extract_modes_all_frames,
    filter_by_energy_percentile,
    mode_centers_from_power,
    mode_energy,
)
from .freq_utils import (
    log_scales,
    rfft_freqs,
    scales_to_freqs,
    stft_times,
)
from .meanshift import ClusterResult, meanshift, meanshift_2d
from .metrics import (
    consistency,
    detection_metrics,
    match_centers,
    reconstruction_rmse,
)
from .reducers import PCA, Precomputed, ProjectionResult
from .windows import derivative_window, make_window

__all__ = [
    # config / errors
    "DAMDConfig", "DAMDError", "ConfigError", "NotFittedError",
    "ConvergenceWarning",
    # aggregation
    "aggregate_power", "aggregate_projected", "project_signal",
    # bandwidth
    "silverman", "adaptive", "percentile", "estimate_bandwidth",
    # clustering
    "meanshift", "meanshift_2d", "ClusterResult",
    # extraction
    "apply_band_mask", "extract_modes_all_frames",
    "mode_centers_from_power", "mode_energy",
    "filter_by_energy_percentile",
    # reducers
    "PCA", "Precomputed", "ProjectionResult",
    # frequency utils
    "rfft_freqs", "log_scales", "scales_to_freqs", "stft_times",
    # metrics
    "consistency", "detection_metrics", "match_centers",
    "reconstruction_rmse",
    # windows
    "make_window", "derivative_window",
]

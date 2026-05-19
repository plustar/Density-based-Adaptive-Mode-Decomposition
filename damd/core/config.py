# damd/core/config.py
"""Configuration for the DAMD pipeline.

A single dataclass that holds every knob used by the pipeline. The
main class :class:`damd.DAMD` accepts either a :class:`DAMDConfig` or
keyword arguments that are forwarded to its constructor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from .exceptions import ConfigError

Transform = Literal["stft", "cwt", "ssq_stft", "ssq_cwt"]
Bandwidth = Literal["silverman", "adaptive", "percentile"]
Space = Literal["projected", "original", "both"]


@dataclass
class DAMDConfig:
    """All parameters of the DAMD pipeline.

    Paper cross-reference
    ---------------------
    - ``transform``: Section II-D of the DAMD paper (eq. 51–53).
    - ``bandwidth``: Section II-C-3 of the DAMD paper (eq. 40, 43, 45).
    - ``r``, ``U``: Definition 3 of the MDAMD paper — forward-backward
      decomposition with projection matrix ``U``.
    - ``characterize``: whether modes are reported in the reduced
      :math:`r`-dimensional space, in the original :math:`d`-dimensional
      space, or both.
    """

    # --- Sampling ---
    fs: float = 1.0                                  # sampling rate (Hz)

    # --- Time-frequency representation ---
    transform: Transform = "stft"
    n_fft: int = 256                                 # STFT only
    hop_length: int = 1                              # STFT only (=1 → full recon)
    window: str = "hann"                             # STFT only
    wavelet: str = "gmw"                             # CWT only
    wavelet_params: dict = field(default_factory=dict)
    n_scales: Optional[int] = None                   # CWT only (None → auto)
    scales_type: str = "log-piecewise"               # CWT only

    # --- Clustering (meanshift) ---
    bandwidth: Bandwidth = "percentile"              # base rule
    bandwidth_scale: float = 1.0                     # global multiplier
    seed_stride: int = 1                             # sub-sampling for seeds
    max_iter: int = 200
    tol: float = 1e-6

    # --- Multivariate / high-dim projection (MDAMD) ---
    r: Optional[int] = None                          # if set: PCA to rank r
    U: Optional[np.ndarray] = None                   # precomputed (d, r), overrides r
    characterize: Space = "projected"                # where to extract modes

    # --- Post-processing ---
    energy_percentile: Optional[float] = None        # drop modes below percentile
    extract_modes: bool = True                       # build hard-mask modes

    # ---------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ConfigError(f"fs must be positive, got {self.fs}")
        if self.transform not in ("stft", "cwt", "ssq_stft", "ssq_cwt"):
            raise ConfigError(f"unknown transform '{self.transform}'")
        if self.bandwidth not in ("silverman", "adaptive", "percentile"):
            raise ConfigError(f"unknown bandwidth rule '{self.bandwidth}'")
        if self.characterize not in ("projected", "original", "both"):
            raise ConfigError(f"unknown characterize mode '{self.characterize}'")
        if self.hop_length != 1 and self.transform.startswith("ssq"):
            raise ConfigError(
                "synchrosqueezed transforms require hop_length=1 "
                "for perfect reconstruction"
            )
        if self.U is not None:
            if self.U.ndim != 2:
                raise ConfigError("U must be 2-D (d, r)")
            self.r = self.U.shape[1]

    # ---------------------------------------------------------------
    def replace(self, **kwargs: Any) -> "DAMDConfig":
        """Return a new config with some fields overridden."""
        from dataclasses import replace
        return replace(self, **kwargs)

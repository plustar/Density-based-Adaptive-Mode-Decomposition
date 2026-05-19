# damd/tfmaps/__init__.py
"""Time-frequency transforms used by the DAMD / MDAMD papers.

Four transforms are supported, each with an inverse:

=========  ======================  ======================
Kind       Forward                 Inverse
=========  ======================  ======================
stft       :func:`stft`            :func:`istft`
cwt        :func:`cwt`             :func:`icwt`
ssq_stft   :func:`ssq_stft`        :func:`issq_stft`
ssq_cwt    :func:`ssq_cwt`         :func:`issq_cwt`
=========  ======================  ======================

:func:`forward` and :func:`inverse` are dispatchers that pick the right
function from a :class:`~damd.DAMDConfig`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import TFMap
from .cwt import cwt, icwt
from .ssq_cwt import issq_cwt, ssq_cwt
from .ssq_stft import issq_stft, ssq_stft
from .stft import istft, stft

__all__ = [
    "TFMap",
    "stft", "istft",
    "cwt", "icwt",
    "ssq_stft", "issq_stft",
    "ssq_cwt", "issq_cwt",
    "forward", "inverse",
]


# ----------------------------------------------------------------------
def forward(x: np.ndarray, config) -> TFMap:
    """Dispatch to the transform requested by ``config.transform``."""
    if config.transform == "stft":
        return stft(
            x, config.fs,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window=config.window,
        )
    if config.transform == "cwt":
        return cwt(
            x, config.fs,
            wavelet=config.wavelet,
            wavelet_params=config.wavelet_params,
            n_scales=config.n_scales,
        )
    if config.transform == "ssq_stft":
        return ssq_stft(
            x, config.fs,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            window=config.window,
        )
    if config.transform == "ssq_cwt":
        return ssq_cwt(
            x, config.fs,
            wavelet=config.wavelet,
            wavelet_params=config.wavelet_params,
            n_scales=config.n_scales,
        )
    raise ValueError(f"unknown transform '{config.transform}'")


def inverse(tf_map: TFMap) -> np.ndarray:
    """Dispatch to the inverse matching ``tf_map.kind``."""
    if tf_map.kind == "stft":
        return istft(tf_map)
    if tf_map.kind == "cwt":
        return icwt(tf_map)
    if tf_map.kind == "ssq_stft":
        return issq_stft(tf_map)
    if tf_map.kind == "ssq_cwt":
        return issq_cwt(tf_map)
    raise ValueError(f"unknown kind '{tf_map.kind}'")

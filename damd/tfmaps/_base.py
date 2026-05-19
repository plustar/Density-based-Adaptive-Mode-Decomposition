# damd/tfmaps/_base.py
"""Time-frequency map container.

Every transform in this subpackage returns a :class:`TFMap`, a simple
immutable dataclass bundling the complex spectrogram with its time and
frequency axes. This keeps the top-level DAMD class agnostic of which
specific transform was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


TransformKind = Literal["stft", "cwt", "ssq_stft", "ssq_cwt"]


@dataclass
class TFMap:
    """A complex time-frequency representation of one or more channels.

    Attributes
    ----------
    tf : (C, F, T) complex ndarray — always 3-D, even for a single channel
    freqs : (F,) frequency axis in Hz
    times : (T,) time axis in seconds
    kind : which transform produced this map
    fs : sampling rate
    meta : free-form dict with transform-specific extras (e.g. ``n_fft``,
        ``scales``) used by the corresponding inverse
    """

    tf: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    kind: TransformKind
    fs: float
    meta: dict = field(default_factory=dict)

    # -- shape helpers --------------------------------------------------
    @property
    def n_channels(self) -> int:
        return self.tf.shape[0]

    @property
    def n_freqs(self) -> int:
        return self.tf.shape[1]

    @property
    def n_times(self) -> int:
        return self.tf.shape[2]

    def power(self) -> np.ndarray:
        """Aggregated power spectrum :math:`\\sum_c |X_c|^2`."""
        return np.sum(np.abs(self.tf) ** 2, axis=0)

    def channel(self, c: int) -> "TFMap":
        """Return a single-channel view."""
        return TFMap(
            tf=self.tf[c : c + 1],
            freqs=self.freqs,
            times=self.times,
            kind=self.kind,
            fs=self.fs,
            meta=self.meta,
        )

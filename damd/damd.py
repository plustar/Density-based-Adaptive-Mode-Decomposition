# damd/damd.py
"""Main entry point: the :class:`DAMD` pipeline class.

One constructor handles all three regimes covered by the two papers:

=========================  ==========================  ================================
Input shape                Config                      Method
=========================  ==========================  ================================
``(N,)``                   any                         single-channel DAMD
``(d, N)``  with ``r=None``                            multivariate DAMD (A = I_d)
``(d, N)``  with ``r=r0``   or ``U=U0``                 projected multivariate DAMD
                                                       (A = U U^T, high-dim case)
=========================  ==========================  ================================

The algorithm follows the forward-backward decomposition of
Definition 3 of the MDAMD paper:

**Forward phase** — optional channel projection → time-frequency
representation of the (reduced) signal → aggregated power spectrum
:math:`P(\\omega,t)` → meanshift partition into frequency bands
:math:`\\{B_k(t)\\}`.

**Backward phase** — mask the time-frequency map with each band to
extract modes, in either the reduced (r-dim) or the original (d-dim)
space, depending on ``characterize``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np

from . import tfmaps
from .core import (
    DAMDConfig,
    PCA,
    Precomputed,
    aggregate_power,
    consistency,
    extract_modes_all_frames,
    filter_by_energy_percentile,
    meanshift_2d,
    mode_centers_from_power,
    mode_energy,
    project_signal,
)
from .core.exceptions import ConfigError, NotFittedError


# ======================================================================
@dataclass
class DAMDResult:
    """Output of :meth:`DAMD.fit`.

    All frame-level fields are *lists* of length ``T`` (number of time
    frames / samples in the output), each item a 1-D or 2-D array whose
    first axis is the variable mode count ``K_t``.

    Attributes
    ----------
    centers : list[ndarray]
        Per-frame power-weighted centre frequencies in Hz. Each item has
        shape ``(K_t,)``.
    bands : list[ndarray]
        Per-frame inclusive frequency-bin bands. Each item has shape
        ``(K_t, 2)``.
    energy : list[ndarray]
        Per-frame, per-mode energy :math:`\\sum_{\\omega\\in B_k}|X|^2`.
    consistency : list[ndarray] or None
        Cross-channel consistency from MDAMD eq. (14); ``None`` for a
        single-channel run.
    modes : list[ndarray] or None
        Per-frame hard-masked time-frequency modes. Populated only when
        ``config.extract_modes=True``. Each item has shape
        ``(K_t, d_char, F)``, where ``d_char`` is ``r`` if
        ``characterize='projected'`` else ``d``.
    freqs : ndarray
        Frequency axis in Hz shared by all frames.
    times : ndarray
        Time axis in seconds of the time-frequency representation.
    power : ndarray
        Aggregated power spectrum ``P(ω, t)`` of shape ``(F, T)``.
    U : ndarray or None
        Projection basis of shape ``(d, r)`` when a reducer was used.
    energy_retention : float or None
        ``η`` from Definition 7 of the MDAMD paper, only when PCA was
        used.
    bandwidth : ndarray
        ``(T,)`` per-frame clustering bandwidth used by meanshift.
    """

    centers: List[np.ndarray]
    bands: List[np.ndarray]
    energy: List[np.ndarray]
    freqs: np.ndarray
    times: np.ndarray
    power: np.ndarray
    bandwidth: np.ndarray
    consistency: Optional[List[np.ndarray]] = None
    modes: Optional[List[np.ndarray]] = None
    U: Optional[np.ndarray] = None
    energy_retention: Optional[float] = None
    kind: str = "stft"
    fs: float = 1.0

    # --- convenience --------------------------------------------------
    def n_modes_per_frame(self) -> np.ndarray:
        """Detected mode count at each time step."""
        return np.array([c.size for c in self.centers], dtype=np.int64)

    def mean_n_modes(self) -> float:
        return float(self.n_modes_per_frame().mean())

    def summary(self) -> str:
        n = self.n_modes_per_frame()
        lines = [
            f"DAMD result — transform={self.kind}, fs={self.fs} Hz",
            f"  grid       : F={self.freqs.size}, T={self.times.size}",
            f"  modes/frame: min={n.min()}, mean={n.mean():.1f}, max={n.max()}",
            f"  bandwidth  : mean={self.bandwidth.mean():.4g}"
            f" (min={self.bandwidth.min():.4g}, max={self.bandwidth.max():.4g})",
        ]
        if self.U is not None:
            lines.append(
                f"  projection : U ∈ R^{self.U.shape[0]}×{self.U.shape[1]}"
                + (f", η={self.energy_retention:.3f}"
                   if self.energy_retention is not None else "")
            )
        if self.consistency is not None:
            mean_c = np.mean([c.mean() if c.size else 0.0
                              for c in self.consistency])
            lines.append(f"  mean C_k   : {mean_c:.3f}")
        return "\n".join(lines)

    def filter_energy(self, percentile: float) -> "DAMDResult":
        """Drop modes whose energy is below the given global percentile.

        All per-frame lists — ``bands``, ``centers``, ``energy`` and
        ``consistency`` — are sliced by the same kept-mode mask so that
        index ``k`` still refers to the same physical mode across all
        four. ``modes`` is invalidated (set to ``None``) because the
        hard-mask tensors would be corrupted by re-indexing.
        """
        # Compute the mask per frame first, then apply it uniformly
        import numpy as np
        all_e = np.concatenate([e for e in self.energy if e.size > 0]) \
            if any(e.size for e in self.energy) else np.array([0.0])
        if all_e.size == 0 or all_e.max() == 0:
            return self
        threshold = float(np.percentile(all_e, percentile))

        kept_b, kept_c, kept_e, kept_cons = [], [], [], []
        for i, (b, c, e) in enumerate(zip(self.bands, self.centers, self.energy)):
            if e.size == 0:
                kept_b.append(b); kept_c.append(c); kept_e.append(e)
                if self.consistency is not None:
                    kept_cons.append(self.consistency[i])
                continue
            keep = e >= threshold
            kept_b.append(b[keep])
            kept_c.append(c[keep])
            kept_e.append(e[keep])
            if self.consistency is not None:
                kept_cons.append(self.consistency[i][keep])

        return DAMDResult(
            centers=kept_c, bands=kept_b, energy=kept_e,
            freqs=self.freqs, times=self.times, power=self.power,
            bandwidth=self.bandwidth,
            consistency=kept_cons if self.consistency is not None else None,
            modes=None,                                   # invalidate
            U=self.U, energy_retention=self.energy_retention,
            kind=self.kind, fs=self.fs,
        )


# ======================================================================
class DAMD:
    """The full DAMD / MDAMD pipeline.

    Examples
    --------
    Single-channel, default settings::

        from damd import DAMD
        res = DAMD(fs=256).fit(x)
        res.summary()

    Multivariate (MDAMD)::

        res = DAMD(fs=256, transform='ssq_stft').fit(X)    # X: (d, N)
        res.consistency              # list[ndarray], one per frame

    High-dimensional projection::

        res = DAMD(fs=256, r=4, characterize='original').fit(X)  # X: (64, N)
        res.U                        # (64, 4) PCA basis
        res.energy_retention         # η (Def 7 MDAMD)

    Pre-computed projection::

        res = DAMD(fs=256, U=my_basis, characterize='projected').fit(X)
    """

    # ------------------------------------------------------------------
    def __init__(self,
                 config: Optional[DAMDConfig] = None,
                 **kwargs: Any):
        if config is not None and kwargs:
            raise ConfigError(
                "pass either a DAMDConfig or keyword arguments, not both"
            )
        self.config: DAMDConfig = config or DAMDConfig(**kwargs)
        self._last: Optional[DAMDResult] = None

    # ------------------------------------------------------------------
    @property
    def result_(self) -> DAMDResult:
        """Output of the most recent :meth:`fit` call."""
        if self._last is None:
            raise NotFittedError("call fit(x) first")
        return self._last

    # ------------------------------------------------------------------
    def fit(self, x: np.ndarray) -> DAMDResult:
        """Run the full pipeline on a signal.

        Parameters
        ----------
        x : (N,) or (d, N) real array.

        Returns
        -------
        :class:`DAMDResult`
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2:
            raise ConfigError(f"expected 1-D or 2-D input, got shape {x.shape}")
        d, N = x.shape
        cfg = self.config

        # ── Forward phase, step 1 : projection -------------------------
        U: Optional[np.ndarray] = None
        energy_retention: Optional[float] = None
        if cfg.U is not None:
            proj = Precomputed(cfg.U).fit(x)
            U = proj.U
        elif cfg.r is not None and cfg.r < d:
            proj = PCA(cfg.r).fit(x)
            U = proj.U
            energy_retention = proj.energy_retention

        if U is not None:
            Y = project_signal(x, U)     # (r, N)
        else:
            Y = x                        # (d, N)

        # ── Forward phase, step 2 : time-frequency transform -----------
        # We project-then-transform (Route A of MDAMD §III Def 3). For
        # linear T, T(U^T X) = U^T T(X) so this is algebraically
        # equivalent to transforming X then aggregating via U U^T; we
        # prefer Route A because it runs the transform on the smaller
        # (r, N) signal. Users wanting the explicit Route B aggregation
        # can call ``damd.core.aggregate_projected(tf, U)`` directly.
        tf_reduced = tfmaps.forward(Y, cfg)            # always 3-D

        # ── Forward phase, step 3 : aggregated power -------------------
        # Because Y is already projected, summing |Y_c|² channel-by-
        # channel is exactly ‖U^T s(ω,t)‖² — the quadratic-form P(ω; U U^T)
        # from Def 2 of the MDAMD paper.
        power = aggregate_power(tf_reduced.tf)         # (F, T)
        freqs = tf_reduced.freqs
        times = tf_reduced.times

        # ── Forward phase, step 4 : meanshift partition per frame ------
        cluster_results = meanshift_2d(
            power, freqs,
            bandwidth_rule=cfg.bandwidth,
            bandwidth_scale=cfg.bandwidth_scale,
            seed_stride=cfg.seed_stride,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
        )
        bands = [cr.bands for cr in cluster_results]
        bw_arr = np.array([cr.bandwidth for cr in cluster_results])

        # Power-weighted centre refinement (eq. 24, DAMD paper)
        centers = mode_centers_from_power(power, freqs, bands)

        # Per-mode energy
        energies = mode_energy(tf_reduced.tf, bands)

        # Optional energy filtering
        if cfg.energy_percentile is not None:
            bands, centers, energies = filter_by_energy_percentile(
                bands, centers, energies, cfg.energy_percentile
            )

        # ── Backward phase : extract modes -----------------------------
        extract_modes = cfg.extract_modes
        tf_for_extract = tf_reduced
        if U is not None and cfg.characterize == "original":
            # Need the transform of the *original* signal
            tf_for_extract = tfmaps.forward(x, cfg)
        modes = None
        if extract_modes:
            modes = extract_modes_all_frames(
                tf_for_extract.tf, bands, keep_complex=True
            )

        # ── Consistency (multi-channel only) ---------------------------
        cons = None
        if tf_for_extract.tf.shape[0] > 1:
            cons = []
            for t, c in enumerate(centers):
                if c.size == 0:
                    cons.append(np.zeros(0))
                    continue
                # Per-frame consistency at the detected centres
                ck = consistency(
                    tf_for_extract.tf[:, :, t : t + 1],
                    c, freqs,
                )
                cons.append(ck)

        result = DAMDResult(
            centers=list(centers),
            bands=list(bands),
            energy=list(energies),
            freqs=freqs,
            times=times,
            power=power,
            bandwidth=bw_arr,
            consistency=cons,
            modes=modes,
            U=U,
            energy_retention=energy_retention,
            kind=cfg.transform,
            fs=cfg.fs,
        )
        self._last = result
        return result

    # ------------------------------------------------------------------
    def fit_transform(self, x: np.ndarray) -> DAMDResult:
        """Alias for :meth:`fit`."""
        return self.fit(x)

    # ------------------------------------------------------------------
    def reconstruct(self, mode_index: Optional[int] = None,
                    frame: Optional[int] = None) -> np.ndarray:
        """Invert the hard-masked modes back to the time domain.

        Parameters
        ----------
        mode_index : int or None
            If given, reconstruct only that mode (reshape-compatible
            across frames); otherwise reconstruct the full signal (sum
            of all modes).
        frame : int or None
            If given together with ``mode_index``, reconstruct a single
            frame's mode only.

        Returns
        -------
        ndarray — time-domain reconstruction.
        """
        res = self.result_
        if res.modes is None:
            raise NotFittedError(
                "fit with extract_modes=True to enable reconstruction"
            )
        cfg = self.config
        kind = res.kind

        # Build a single time-frequency map by summing over the modes
        # requested, then call the matching inverse.
        T = len(res.modes)
        # Infer (d, F) from the first non-empty frame
        ref = next((m for m in res.modes if m.size), None)
        if ref is None:
            raise RuntimeError("no modes were detected in any frame")
        _, d, F = ref.shape
        tf = np.zeros((d, F, T), dtype=complex)
        for t, M in enumerate(res.modes):
            if M.size == 0:
                continue
            if mode_index is not None and frame is not None:
                if t != frame or mode_index >= M.shape[0]:
                    continue
                tf[:, :, t] = M[mode_index]
            elif mode_index is not None:
                if mode_index < M.shape[0]:
                    tf[:, :, t] = M[mode_index]
            else:
                tf[:, :, t] = M.sum(axis=0)

        # Wrap as a TFMap so the inverse can find its meta-data
        from .tfmaps._base import TFMap
        # Borrow meta from a fresh forward on the last fitted signal
        # isn't accessible; reuse the forward cached in self.
        # Simpler: re-run the forward transform on a dummy zero signal
        # just to harvest the meta fields. We reconstruct the meta from
        # the result directly:
        meta = self._guess_meta(res, d)
        map_ = TFMap(tf=tf, freqs=res.freqs, times=res.times,
                     kind=kind, fs=res.fs, meta=meta)
        return tfmaps.inverse(map_)

    # ------------------------------------------------------------------
    def _guess_meta(self, res: DAMDResult, d: int) -> dict:
        """Best-effort reconstruction of the TFMap.meta dict from config."""
        cfg = self.config
        if res.kind in ("stft", "ssq_stft"):
            from .core.windows import make_window
            w = make_window(cfg.window, cfg.n_fft)
            meta = {
                "n_fft": cfg.n_fft,
                "hop_length": 1,
                "window": w,
                "center": True,
                "modulated": True,
                "n_samples": res.times.size,
            }
            if res.kind == "ssq_stft":
                meta["window_center_value"] = float(w[cfg.n_fft // 2])
            return meta
        if res.kind in ("cwt", "ssq_cwt"):
            from .tfmaps._wavelets import wavelet_center_freq
            from .core.freq_utils import log_scales
            params = cfg.wavelet_params
            omega0 = wavelet_center_freq(cfg.wavelet, params)
            # Rebuild scales from the frequency axis
            scales = (cfg.fs * omega0) / (2.0 * np.pi * res.freqs)
            return {
                "wavelet": cfg.wavelet,
                "wavelet_params": params,
                "scales": scales,
                "omega0": omega0,
                "n_samples": res.times.size,
            }
        return {}

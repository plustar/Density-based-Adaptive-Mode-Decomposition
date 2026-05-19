# experiments/mdamd_paper/exp_b_equivalence.py
"""Reproduces §VIII-B / Table IV of the MDAMD paper.

Validates:
* Theorem 1 (MVMD-Aggregation Equivalence) — detection MAE unchanged
  between C=1 single-channel and C=4 aggregation
* Proposition 3 (consistency bounds) — coherent mode at C≈1, incoherent
  at C≈1/C
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from damd import DAMD
from damd.core import match_centers
from experiments.common import apply_style, multichannel_signal


OUT = Path(__file__).parent / "figures"


def _run(seed: int) -> dict:
    fs = 256.0
    X, _, truth = multichannel_signal(fs=fs, duration=4.0,
                                      n_channels=4, seed=seed)
    f_shared = truth["shared_freq"]

    # Single-channel DAMD on first channel (Prop 2 reference)
    res_1 = DAMD(fs=fs, transform="stft", n_fft=128).fit(X[0])
    # Multi-channel MDAMD aggregating all four
    res_C = DAMD(fs=fs, transform="stft", n_fft=128).fit(X)

    # MAE against ground-truth shared frequency
    mae_1 = np.mean([
        np.min(np.abs(c - f_shared)) if c.size else np.nan
        for c in res_1.centers if c.size
    ])
    mae_C = np.mean([
        np.min(np.abs(c - f_shared)) if c.size else np.nan
        for c in res_C.centers if c.size
    ])

    # Consistency: mean C_k at the detected centre closest to the shared freq
    def _coh_at_shared(res):
        if res.consistency is None:
            return np.nan
        vals = []
        for c, cons in zip(res.centers, res.consistency):
            if c.size == 0:
                continue
            k = np.argmin(np.abs(c - f_shared))
            vals.append(cons[k])
        return float(np.mean(vals))

    # Incoherent: pick a centre FAR from the shared frequency
    def _coh_incoh(res):
        if res.consistency is None:
            return np.nan
        vals = []
        for c, cons in zip(res.centers, res.consistency):
            if c.size < 2:
                continue
            far = np.argmax(np.abs(c - f_shared))
            vals.append(cons[far])
        return float(np.mean(vals))

    return {
        "mae_C1": mae_1,
        "mae_C4": mae_C,
        "coh": _coh_at_shared(res_C),
        "incoh": _coh_incoh(res_C),
    }


def main() -> None:
    apply_style()
    rows = [_run(s) for s in range(10)]
    mae_1 = np.array([r["mae_C1"] for r in rows])
    mae_C = np.array([r["mae_C4"] for r in rows])
    coh = np.array([r["coh"] for r in rows])
    inc = np.array([r["incoh"] for r in rows])

    print("=== Table IV: Equivalence validation (10 seeds) ===")
    print(f"  Prop 2  (C=1): MAE = {np.nanmean(mae_1):.3f} ± {np.nanstd(mae_1):.3f} Hz")
    print(f"  Thm 1   (C=4): MAE = {np.nanmean(mae_C):.3f} ± {np.nanstd(mae_C):.3f} Hz")
    print(f"  Coherent   C_k = {np.nanmean(coh):.3f} ± {np.nanstd(coh):.3f}  (expected ≈ 1)")
    print(f"  Incoherent C_k = {np.nanmean(inc):.3f} ± {np.nanstd(inc):.3f}  (expected ≈ 1/C = 0.25)")


if __name__ == "__main__":
    main()

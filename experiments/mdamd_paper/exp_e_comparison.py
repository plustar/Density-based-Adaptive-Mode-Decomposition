# experiments/mdamd_paper/exp_e_comparison.py
"""Reproduces Table VII of the MDAMD paper.

Compares three methods on the multi-channel signal:
* DAMD       (adaptive K, clustering-based)
* MVMD       (fixed K, shared centres)
* STVMD      (fixed K, time-varying centres)

Reports per-method precision / recall / F1 / MAE and wall time.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from damd import DAMD
from damd.baselines import mvmd, stvmd
from damd.core import detection_metrics
from experiments.common import apply_style, multichannel_signal


def _frame_truth(truth: dict, n_frames: int) -> list:
    """Ground-truth centres per frame — we use static truths here."""
    shared = [truth["shared_freq"]]
    specifics = list(truth["channel_freqs"])
    all_truth = np.array(sorted(shared + specifics), dtype=float)
    return [all_truth.copy() for _ in range(n_frames)]


def _expand_static(centers: np.ndarray, n_frames: int) -> list:
    """Repeat a static centre vector across frames for metric computation."""
    return [centers.copy() for _ in range(n_frames)]


def _damd_metrics(X, fs, truth):
    t0 = time.perf_counter()
    res = DAMD(fs=fs, transform="stft", n_fft=128).fit(X)
    dt = time.perf_counter() - t0
    gt = _frame_truth(truth, len(res.centers))
    m = detection_metrics(res.centers, gt, tolerance=2.0)
    m["time_s"] = dt
    return m


def _mvmd_metrics(X, fs, truth, K):
    t0 = time.perf_counter()
    res = mvmd(X, K=K, fs=fs, alpha=2000, max_iter=150)
    dt = time.perf_counter() - t0
    # MVMD gives one (K,) centre vector shared across all time — expand
    # to per-frame for uniform metric
    det = _expand_static(res.centers, 256)
    gt = _frame_truth(truth, 256)
    m = detection_metrics(det, gt, tolerance=2.0)
    m["time_s"] = dt
    return m


def _stvmd_metrics(X, fs, truth, K):
    t0 = time.perf_counter()
    res = stvmd(X, K=K, fs=fs, window_length=128, hop_length=32,
                dynamic=True, alpha=1500, max_iter=40)
    dt = time.perf_counter() - t0
    # STVMD gives (K, T) — convert to per-frame list
    det = [res.centers[:, t] for t in range(res.centers.shape[1])]
    gt = _frame_truth(truth, res.centers.shape[1])
    m = detection_metrics(det, gt, tolerance=2.0)
    m["time_s"] = dt
    return m


def main() -> None:
    apply_style()
    fs = 256.0

    print()
    print("=== Table VII: Detection performance comparison ===")
    header = f"{'Method':<10} | {'C':>3} | {'P':>5} | {'R':>5} | {'F1':>5} | " \
             f"{'MAE':>7} | {'Time (s)':>9}"
    print(header)
    print("-" * len(header))

    for C in (1, 4):
        rows = {"DAMD": [], "MVMD": [], "STVMD": []}
        for seed in range(3):
            X, _, truth = multichannel_signal(fs=fs, duration=2.0,
                                              n_channels=C, seed=seed)
            K_est = len(truth["channel_freqs"]) + 2       # MVMD/STVMD get K_truth + 2

            rows["DAMD"].append(_damd_metrics(X, fs, truth))
            rows["MVMD"].append(_mvmd_metrics(X, fs, truth, K_est))
            rows["STVMD"].append(_stvmd_metrics(X, fs, truth, K_est))

        for name in ("DAMD", "MVMD", "STVMD"):
            xs = rows[name]
            p = np.mean([r["precision"] for r in xs])
            rr = np.mean([r["recall"] for r in xs])
            f1 = np.mean([r["f1"] for r in xs])
            mae = np.nanmean([r["mae"] for r in xs])
            t = np.mean([r["time_s"] for r in xs])
            print(f"{name:<10} | {C:>3} | {p:>5.2f} | {rr:>5.2f} | "
                  f"{f1:>5.2f} | {mae:>7.2f} | {t:>9.2f}")


if __name__ == "__main__":
    main()

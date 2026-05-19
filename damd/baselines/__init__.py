# damd/baselines/__init__.py
"""Traditional VMD-family baselines used for comparison.

All three share the same ADMM core (:mod:`damd.baselines._admm`):

* :func:`vmd`   — single-channel VMD (Dragomiretskiy & Zosso 2014)
* :func:`mvmd`  — multi-channel MVMD (Rehman & Aftab 2019)
* :func:`stvmd` — short-time / dynamic STVMD (Jia et al. 2026)
"""
from __future__ import annotations

from .mvmd import MVMDResult, mvmd
from .stvmd import STVMDResult, stvmd
from .vmd import VMDResult, vmd

__all__ = [
    "vmd", "VMDResult",
    "mvmd", "MVMDResult",
    "stvmd", "STVMDResult",
]

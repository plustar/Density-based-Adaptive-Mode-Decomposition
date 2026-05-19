# damd/__init__.py
"""damd — Density-based Adaptive Mode Decomposition.

Unified frequency-domain partitioning framework for single-channel,
multi-channel and high-dimensional signals. One class,
:class:`DAMD`, handles all three regimes through the forward-backward
decomposition of Definition 3 of the MDAMD paper.

Quick start::

    import numpy as np
    from damd import DAMD

    x = np.random.randn(1024)          # a single-channel signal
    res = DAMD(fs=256).fit(x)
    print(res.summary())

Citations::

    @article{jia2026damd,
      title  = {Density-based Adaptive Mode Decomposition},
      author = {Jia, H. and Yang, C. and Caiafa, C. F. and
                Sun, Z. and Duan, F. and Sole-Casals, J.},
      journal = {IEEE Transactions on Signal Processing},
      year = {2026}
    }

    @article{jia2026mdamd,
      title  = {Multivariate Density-based Adaptive Mode Decomposition},
      author = {Jia, H. and Yang, C. and Feng, F. and Caiafa, C. F. and
                Duan, F. and Sole-Casals, J.},
      year   = {2026},
      note   = {under submission}
    }
"""
from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Hao Jia"
__email__ = "haojia@nankai.edu.cn"
__affiliation__ = "School of Medicine, Nankai University"

# Public configuration and result
from .core import DAMDConfig
from .damd import DAMD, DAMDResult

# Subpackages exposed as namespaces
from . import baselines, core, tfmaps, viz
from . import hvr as _hvr_module
from .hvr import HVRResult, hvr

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__affiliation__",
    "DAMD",
    "DAMDConfig",
    "DAMDResult",
    "hvr",
    "HVRResult",
    "core",
    "tfmaps",
    "baselines",
    "viz",
]

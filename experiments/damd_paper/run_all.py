# experiments/damd_paper/run_all.py
"""Run every experiment in this folder, in paper order."""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path


EXPERIMENTS = [
    "exp_a_decomposition",
    "exp_b_bandwidth",
    "exp_c_transforms",
    "exp_d_hvr",
]


def main() -> None:
    here = Path(__file__).parent
    sys.path.insert(0, str(here.parent.parent))

    for name in EXPERIMENTS:
        mod_name = f"experiments.damd_paper.{name}"
        print(f"\n{'=' * 70}\n  running {mod_name}\n{'=' * 70}")
        try:
            mod = importlib.import_module(mod_name)
            mod.main()
        except Exception:
            traceback.print_exc()
            print(f"[skipped {name} after error]")


if __name__ == "__main__":
    main()

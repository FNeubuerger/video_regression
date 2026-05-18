"""One-batch smoke test for every LOSO model.

Walks through the same construction code path as ``loso_cross_validation.py``
but only runs a single forward+backward+test step per model.  Prints
``OK`` or the full traceback for each.  Lets us discover all shape /
loss bugs at once instead of one-at-a-time.

Run from the repo root::

    .venv/bin/python tests/smoke_loso_models.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.loso_cross_validation import run_loso_fold  # noqa: E402

# Monkey-patch: limit Subset sizes so smoke test runs in minutes, not hours.
import evaluation.loso_cross_validation as _loso  # noqa: E402
from torch.utils.data import Subset as _Subset  # noqa: E402

_orig_subset = _Subset

class _CappedSubset(_Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, list(indices)[:64])

_loso.Subset = _CappedSubset  # type: ignore[attr-defined]

MODELS = [
    "CNNLSTM",
    "PretrainedCNNLSTM",
    "SimpleResNet",
    "SpatialResNet",
    "PhysicsCNNLSTM",
    "ConvectionBioheat",
    "BayesianResNet",
    "FullBayesianResNet",
    "BayesianCNNLSTM",
    "ConvLTC",
]


class Args:
    epochs = 1
    batch_size = 4
    masked = False


def main():
    # Pick the first available video as the held-out fold so all models
    # share the same tiny train/val/test split.
    import glob

    vids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob("data/level1_cropped/*.mp4")
    )
    if not vids:
        print("No videos found under data/level1_cropped — aborting smoke test.")
        return 1
    holdout = vids[0]
    print(f"Smoke holdout = {holdout}\n")

    results = []
    for m in MODELS:
        print(f"=== {m} ===")
        try:
            res = run_loso_fold(holdout, m, Args)
            status = f"OK   mae={res['mae']:.2f}" if res else "SKIP"
        except Exception as e:  # noqa: BLE001
            status = f"FAIL {type(e).__name__}: {e}"
            traceback.print_exc(limit=5)
        results.append((m, status))
        print()

    print("\n=== SMOKE SUMMARY ===")
    for m, s in results:
        print(f"  {m:25s} {s}")
    failed = [m for m, s in results if s.startswith("FAIL")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

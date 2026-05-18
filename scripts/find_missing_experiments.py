"""Detect which experiment checkpoints are missing.

Prints a colour-free, machine-readable list of model names that have NO
checkpoint on disk.  Driven by ``benchmarks/expected_experiments.txt``
(one model name per line) and the contents of ``models/`` and
``checkpoints/``.

Used by ``make missing`` to drive only the still-pending training runs.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List

CHECKPOINT_PATTERNS: Dict[str, List[str]] = {
    "CNNLSTM": ["models/cnnlstm_model.pth"],
    "PretrainedCNNLSTM": ["models/pretrained_cnnlstm_model.pth"],
    "SimpleResNet": ["models/simple_resnet_model.pth"],
    "PhysicsCNNLSTM": ["models/physics_cnnlstm_model.pth"],
    "SpatialBioheat": ["models/spatial_bioheat_resnet.pth"],
    "SpatialConvection": ["models/spatial_convection_bioheat_resnet.pth"],
    "SpatialMetabolic": ["models/spatial_metabolic_bioheat_resnet.pth"],
    "BioheatPINN": ["models/bioheat_pinn_model.pth"],
    "ConvectionBioheat": ["models/convection_bioheat_model.pth"],
    "MetabolicBioheat": ["models/metabolic_bioheat_model.pth"],
    "BayesianResNet": ["checkpoints/bayesian_resnet.pth"],
    "FullBayesianResNet": ["checkpoints/full_bayesian_resnet.pth"],
    "BayesianCNNLSTM": ["models/bayesian_cnnlstm.pth"],
    "BayesianPINN": ["models/bayesian_pinn.pth"],
    "ConvLTC": ["checkpoints/conv_ltc/*.pth"],
    "LatentLTC_UNet": ["checkpoints/ltc_unet/*.pth"],
    "ResNetUNet": [
        "checkpoints/unet_hybrid/*.pth",
        "checkpoints/unet_sparse/*.pth",
    ],
    "Ensemble": ["checkpoints/ensemble/model_0.pth"],
}


def find_missing() -> List[str]:
    missing = []
    for name, patterns in CHECKPOINT_PATTERNS.items():
        if not any(glob.glob(p) for p in patterns):
            missing.append(name)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-only", action="store_true", help="One model per line.")
    args = ap.parse_args()

    miss = find_missing()
    if args.list_only:
        for m in miss:
            print(m)
        return 0

    if not miss:
        print("All registered checkpoints are present.")
        return 0
    print(f"Missing {len(miss)} checkpoint(s):")
    for m in miss:
        print(f"  - {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

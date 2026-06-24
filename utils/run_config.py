"""Persist run provenance next to every checkpoint / log directory.

Writes a single ``run_config.json`` summarising the seed, hyperparameters,
git commit, hardware, library versions and command line.  This is the
information reviewers and future-us need to reproduce a result.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        return bool(out.strip())
    except Exception:
        return False


def _gpu_info() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "capability": [
            list(torch.cuda.get_device_capability(i))
            for i in range(torch.cuda.device_count())
        ],
        "cuda_version": torch.version.cuda,
    }


@dataclass
class RunConfig:
    model: str
    seed: int
    hparams: Dict[str, Any] = field(default_factory=dict)
    data_dir: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "seed": self.seed,
            "hparams": self.hparams,
            "data_dir": self.data_dir,
            "notes": self.notes,
            "git_sha": _git_sha(),
            "git_dirty": _git_dirty(),
            "torch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "gpu": _gpu_info(),
            "argv": sys.argv,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }


def write_run_config(out_dir: str, cfg: RunConfig) -> str:
    """Write ``run_config.json`` inside ``out_dir`` and return the path."""

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2, default=str)
    return path

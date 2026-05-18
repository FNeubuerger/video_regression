"""Global determinism / seeding utilities.

Use ``set_global_seed`` at the very top of every training and evaluation
entrypoint so results can be reproduced exactly.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed every relevant RNG and (optionally) force deterministic cuDNN.

    Parameters
    ----------
    seed:
        Integer seed used for ``random``, ``numpy``, ``torch`` and
        ``torch.cuda``.  Also exported as ``PYTHONHASHSEED``.
    deterministic:
        When True we force cuDNN into deterministic mode.  This usually slows
        training down but is required for bit-reproducible results.

    Returns
    -------
    int
        The seed that was applied (handy for logging).
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that derives a deterministic per-worker seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

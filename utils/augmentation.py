"""Lightweight train-time augmentation for LOSO sequence training.

Only used on the training split (never val/test) to reduce subject-specific
overfitting without touching the shared training loop in
``training/train_all_models.py``.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


class AugmentedSequenceSubset(Dataset):
    """Wraps a ``SequenceHeatmapDataset`` Subset with random flip + brightness jitter.

    Each sample is a tuple whose first element is the ``(T, C, H, W)`` image
    tensor (RGB channels 0-2, optical-flow dx/dy channels 3-4 if present).
    Horizontal flips mirror every spatial tensor in the tuple (heatmap
    targets, artifact masks) and negate the flow-dx channel so the flow
    field stays physically consistent after mirroring.
    """

    def __init__(
        self, subset: Dataset, flip_prob: float = 0.5, jitter_strength: float = 0.1
    ):
        self.subset = subset
        self.flip_prob = flip_prob
        self.jitter_strength = jitter_strength

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx):
        item = list(self.subset[idx])
        imgs = item[0].clone()

        if random.random() < self.flip_prob:
            imgs = torch.flip(imgs, dims=[-1])
            if imgs.shape[1] >= 4:
                imgs[:, 3, :, :] = -imgs[:, 3, :, :]  # negate flow-dx (x-component)
            for j in range(1, len(item)):
                t = item[j]
                if (
                    isinstance(t, torch.Tensor)
                    and t.dim() >= 3
                    and t.shape[-1] == imgs.shape[-1]
                ):
                    item[j] = torch.flip(t, dims=[-1])

        if self.jitter_strength > 0:
            brightness = 1.0 + (random.random() * 2 - 1) * self.jitter_strength
            imgs[:, :3] = (imgs[:, :3] * brightness).clamp(min=-5.0, max=5.0)

        item[0] = imgs
        return tuple(item)

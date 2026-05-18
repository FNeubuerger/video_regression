"""Helpers to load a trained dense-temperature model and run it on a video.

Used by ``validate_ablation_zone.py`` to replace the grayscale-stub
``predict_temperature_maps`` with a real network forward pass.

The function returns a ``(T, H, W)`` numpy array of temperatures in
Celsius and, for uncertainty-aware models, an optional matching
``(T, H, W)`` array of per-pixel standard deviations.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from utils.model_registry import MODEL_REGISTRY


def _read_video_as_tensor(
    video_path: str, target_size: Tuple[int, int] = (64, 64)
) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    arr = np.stack(frames, axis=0)  # (T, H, W, 3)
    arr = arr.transpose(0, 3, 1, 2)  # (T, 3, H, W)
    return torch.from_numpy(arr)


def _maybe_pad_flow(rgb: torch.Tensor) -> torch.Tensor:
    """Pad 3-channel RGB with 2 zero flow channels to match training inputs."""
    if rgb.shape[1] == 5:
        return rgb
    pad = torch.zeros(rgb.shape[0], 2, rgb.shape[2], rgb.shape[3])
    return torch.cat([rgb, pad], dim=1)


def _windowed(frames: torch.Tensor, window: int = 5):
    """Yield sliding windows along the time axis."""
    T = frames.shape[0]
    for end in range(window, T + 1):
        yield end - 1, frames[end - window : end]


def load_model(model_name: str, checkpoint_path: str, device: str = "cuda"):
    if model_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model {model_name} not registered. Available: {list(MODEL_REGISTRY)}"
        )
    cls, kwargs = MODEL_REGISTRY[model_name]
    model = cls(**kwargs)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def predict_video_temperatures(
    model: torch.nn.Module,
    video_path: str,
    device: str = "cuda",
    target_size: Tuple[int, int] = (64, 64),
    window: int = 5,
    num_mc_samples: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return ``(T, H, W)`` temperature maps and optional std maps.

    For Bayesian / dropout models pass ``num_mc_samples > 1`` to enable
    Monte Carlo predictive distributions.  Spatial models are expected to
    output a tensor that can be reduced to ``(B, H, W)`` per call.
    """

    rgb = _read_video_as_tensor(video_path, target_size=target_size)
    rgb = _maybe_pad_flow(rgb)  # (T, 5, H, W)
    T_total = rgb.shape[0]
    H, W = rgb.shape[-2:]

    temp_maps = np.zeros((T_total, H, W), dtype=np.float32)
    std_maps = np.zeros((T_total, H, W), dtype=np.float32) if num_mc_samples > 1 else None

    # Use sliding windows; fill prefix by repeating the first valid frame.
    last_t_pred = None
    for end_idx, win in _windowed(rgb, window=window):
        batch = win.unsqueeze(0).to(device)  # (1, T, C, H, W)
        samples = []
        for _ in range(max(num_mc_samples, 1)):
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            # Squeeze to (H, W).
            if out.dim() == 5:
                out = out[:, -1]
            if out.dim() == 4:
                out = out.mean(dim=1) if out.shape[1] > 1 else out[:, 0]
            if out.dim() == 3:
                out = out[0]
            samples.append(out.cpu().numpy())
        s = np.stack(samples, axis=0)
        temp_maps[end_idx] = s.mean(axis=0)
        if std_maps is not None:
            std_maps[end_idx] = s.std(axis=0)
        last_t_pred = temp_maps[end_idx]

    # Forward-fill the first ``window-1`` frames so we have a full sequence.
    if last_t_pred is not None:
        first = temp_maps[window - 1]
        for i in range(window - 1):
            temp_maps[i] = first

    return temp_maps, std_maps


def predict_with_ablation_probability(
    model: torch.nn.Module,
    video_path: str,
    cem43_threshold: float,
    num_mc_samples: int = 16,
    device: str = "cuda",
    dt_minutes: float = 1.0 / 30,
) -> np.ndarray:
    """Probabilistic ablation map: ``P(CEM43 > threshold)`` per pixel.

    Runs ``num_mc_samples`` stochastic forward passes, computes a CEM43
    map per sample and returns the empirical exceedance probability.
    """
    from validation.validate_ablation_zone import calculate_cem43  # local import

    probs = None
    for _ in range(num_mc_samples):
        temp_seq, _ = predict_video_temperatures(
            model, video_path, device=device, num_mc_samples=1
        )
        cem = calculate_cem43(temp_seq, dt=dt_minutes)
        hit = (cem > cem43_threshold).astype(np.float32)
        probs = hit if probs is None else probs + hit
    return probs / num_mc_samples

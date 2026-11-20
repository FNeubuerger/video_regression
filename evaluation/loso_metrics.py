"""Strict, last-frame thermometry metrics for held-out ultrasound windows."""

import torch
import torch.nn.functional as F


def batch_errors(model, batch, device, *, spatial, mc_samples=1):
    """Return four-probe mean, partial-probe mean, and individual-probe errors.

    The dataset has already masked the video. Its third tensor is a temperature
    prior, not an input mask. Keep partial windows out of the primary metric.
    """
    images, heatmaps, _prior, sensors, measured = batch
    b, t = images.shape[:2]
    if sensors.shape != (b, t, 4):
        raise ValueError(f"Expected (B, T, 4) sensor readings, got {sensors.shape}")
    images = images.to(device)
    samples = []
    for _ in range(mc_samples):
        raw = model(images)
        samples.append(raw[0] if isinstance(raw, tuple) else raw)
    prediction = torch.stack(samples).mean(dim=0).cpu()
    if not torch.isfinite(prediction).all():
        raise ValueError("Non-finite LOSO predictions")
    target = sensors[:, -1]
    valid = torch.isfinite(target)
    if spatial:
        if prediction.ndim == 5 and prediction.shape[:3] == (b, t, 1):
            prediction = prediction[:, -1, 0]
        elif prediction.ndim == 4 and prediction.shape[:2] == (b, t):
            prediction = prediction[:, -1]
        if prediction.ndim != 3 or prediction.shape[0] != b:
            raise ValueError(f"Unexpected spatial output shape: {prediction.shape}")
        if heatmaps.shape != measured.shape or heatmaps.shape[:3] != (b, t, 1):
            raise ValueError("Expected matching (B, T, 1, H, W) maps and probe masks")
        observed = measured[:, -1, 0]
        upsampled = F.interpolate(
            prediction[:, None],
            size=heatmaps.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        if not torch.equal(observed.sum(dim=(1, 2)), valid.sum(dim=1)):
            raise ValueError("Sensor readings and mapped probe counts differ")
        differences = upsampled - heatmaps[:, -1, 0]
        probe_errors = differences[observed]
        sums = differences.masked_fill(~observed, 0).sum(dim=(1, 2))
    else:
        if prediction.shape == (b, t, 4):
            prediction = prediction[:, -1]
        if prediction.shape == (b, 4):
            differences = prediction - target
            probe_errors = differences[valid]
            sums = differences.masked_fill(~valid, 0).sum(dim=1)
        elif prediction.shape in {(b,), (b, 1)}:
            complete = valid.all(dim=1)
            return (
                prediction.reshape(b)[complete] - target[complete].mean(dim=1),
                prediction.new_empty(0),
                prediction.new_empty(0),
            )
        else:
            raise ValueError(f"Unexpected temperature output shape: {prediction.shape}")
    counts = valid.sum(dim=1)
    complete = counts == 4
    partial = (counts > 0) & ~complete
    return sums[complete] / 4, sums[partial] / counts[partial], probe_errors

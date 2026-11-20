"""LOSO training for deterministic four-probe regression heads."""

import numpy as np
import torch


def split_by_sequence(windows, seed):
    """Partition (sequence ID, dataset index) pairs without sharing sequences."""
    groups = sorted({group for group, _ in windows})
    if len(groups) < 2:
        raise ValueError("LOSO fitting needs at least two non-test sequences")
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(groups)[: max(1, round(0.15 * len(groups)))].tolist())
    train = [idx for group, idx in windows if group not in held]
    val = [idx for group, idx in windows if group in held]
    return train, val, sorted(held)


def measured_probe_errors(prediction, sensors):
    """Pair the four outputs with measured final-frame readings only."""
    if sensors.ndim != 3 or sensors.shape[1] == 0 or sensors.shape[2] != 4:
        raise ValueError(f"Expected (B, T, 4) sensor readings, got {sensors.shape}")
    target = sensors[:, -1]
    if prediction.shape != target.shape:
        raise ValueError(
            f"Expected {target.shape} probe outputs, got {prediction.shape}"
        )
    if not torch.isfinite(prediction).all():
        raise ValueError("Non-finite probe predictions")
    measured = torch.isfinite(target)
    return prediction[measured] - target[measured]


def _epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_sse, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, _maps, _prior, sensors, _mask in loader:
            sensors = sensors.to(device)
            # Do not let unlabelled windows update the model's batch statistics.
            labelled = torch.isfinite(sensors[:, -1]).any(dim=1)
            if not labelled.any():
                continue
            errors = measured_probe_errors(
                model(images.to(device)[labelled]), sensors[labelled]
            )
            sse = errors.square().sum()
            if not torch.isfinite(sse):
                raise ValueError("Non-finite probe loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (sse / errors.numel()).backward()
                optimizer.step()
            total_sse += sse.detach().item()
            count += errors.numel()
    if not count:
        raise ValueError("No measured probes in the training or validation split")
    return total_sse / count


def fit_probe_model(model, train, val, optimizer, device, epochs, save_path):
    """Select the checkpoint on validation probe MSE, never on the test fold."""
    model.to(device)
    best_loss, best_epoch = float("inf"), 0
    for epoch in range(1, epochs + 1):
        train_loss = _epoch(model, train, device, optimizer)
        val_loss = _epoch(model, val, device)
        print(
            f"Epoch {epoch}/{epochs}: probe MSE train={train_loss:.4f} val={val_loss:.4f}"
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), save_path)
        elif epoch - best_epoch >= 5:
            break
    return best_epoch, best_loss

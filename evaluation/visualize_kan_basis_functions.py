"""Visualize learned KAN basis functions for interpretability & diagnosis.

Loads trained KAN checkpoints and plots the univariate functions learned
by each KANLinear layer. This helps diagnose pathological behavior (e.g.,
exploding gradients, saturation) and demonstrate KAN interpretability.

Usage:
    .venv/bin/python evaluation/visualize_kan_basis_functions.py \
        --model KANResNet \
        --checkpoint checkpoints/loso/KANResNet/fold_US_001_30W_10min.pth \
        --output paper/figures/kan_basis_functions_kanresnet.png
"""

from __future__ import annotations

import argparse
import os
import sys

# Allow `import models.*` when invoked from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.kan import KAN, KANLinear


def extract_kan_linear_functions(
    kan_linear: KANLinear,
    x_range: tuple[float, float] = (-1.0, 1.0),
    n_points: int = 100,
) -> dict[str, np.ndarray]:
    """Extract learned univariate functions from a KANLinear layer.

    Returns:
        {
            "x": (n_points,) input samples,
            "y": (n_points,) output for a canonical input,
            "y_base": (n_points,) base activation output,
            "y_spline": (n_points,) spline activation output,
        }
    """
    x = torch.linspace(x_range[0], x_range[1], n_points, dtype=torch.float32)
    x_expanded = x.unsqueeze(-1).repeat(1, kan_linear.in_features)  # (n_points, in_features)
    x_expanded = x_expanded.to(next(kan_linear.parameters()).device)

    with torch.no_grad():
        # Base activation
        base_activated = kan_linear.base_activation(x_expanded)  # (n_points, in_features)
        y_base = (base_activated * kan_linear.scale_base).sum(dim=1)  # (n_points,)

        # Spline activation
        basis = kan_linear._basis(x_expanded)  # (n_points, in_features, n_basis)
        # Contract with spline weights
        weight = kan_linear.spline_weight.reshape(kan_linear.out_features, -1)
        basis_flat = basis.reshape(basis.shape[0], -1)
        y_spline = torch.nn.functional.linear(basis_flat, weight).sum(dim=1)  # (n_points,)

        y = y_base + y_spline

    return {
        "x": x.cpu().numpy(),
        "y": y.cpu().numpy(),
        "y_base": y_base.cpu().numpy(),
        "y_spline": y_spline.cpu().numpy(),
    }


def diagnose_kan_functions(
    model: torch.nn.Module,
    n_samples: int = 100,
) -> dict:
    """Diagnose health of KAN functions in a model.

    Returns diagnosis dict with info about learned functions.
    """
    functions = []
    all_y = []

    # Find all KANLinear layers recursively
    for name, module in model.named_modules():
        if isinstance(module, KANLinear):
            funcs = extract_kan_linear_functions(module, n_points=n_samples)
            functions.append({
                "layer_name": name,
                **funcs,
            })
            all_y.extend(funcs["y"])

    all_y = np.array(all_y)
    is_exploding = np.any(~np.isfinite(all_y))
    max_val = float(np.nanmax(all_y)) if len(all_y) > 0 else 0.0
    min_val = float(np.nanmin(all_y)) if len(all_y) > 0 else 0.0
    is_saturated = (np.abs(all_y) > 100).sum() / len(all_y) > 0.5 if len(all_y) > 0 else False

    return {
        "functions": functions,
        "max_val": max_val,
        "min_val": min_val,
        "mean_magnitude": float(np.nanmean(np.abs(all_y))) if len(all_y) > 0 else 0.0,
        "n_functions": len(functions),
        "is_exploding": is_exploding,
        "is_saturated": is_saturated,
    }


def plot_basis_functions(
    model: torch.nn.Module,
    output_path: str,
    max_functions_to_plot: int = 4,
) -> None:
    """Plot learned KAN basis functions from model."""
    diagnosis = diagnose_kan_functions(model, n_samples=200)

    functions = diagnosis["functions"]
    if not functions:
        print("No KAN functions found in model.")
        return

    # Create plot with one row per KANLinear layer
    n_layers = len(functions)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 3.5 * n_layers))
    if n_layers == 1:
        axes = [axes]

    for layer_idx, func_data in enumerate(functions):
        ax = axes[layer_idx]
        x = func_data["x"]
        y = func_data["y"]
        y_base = func_data["y_base"]
        y_spline = func_data["y_spline"]

        ax.plot(x, y, label="Total (base + spline)", linewidth=2, color="#1f77b4")
        ax.plot(x, y_base, linestyle="--", label="Base activation", linewidth=1.2, color="#ff7f0e")
        ax.plot(x, y_spline, linestyle=":", label="Spline basis", linewidth=1.2, color="#2ca02c")

        ax.set_title(f"KANLinear Layer {layer_idx}: {func_data['layer_name']}")
        ax.set_xlabel("Normalized input ([-1, 1])")
        ax.set_ylabel("Learned function output")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        
        # Add text summary
        if np.any(~np.isfinite(y)):
            ax.text(0.02, 0.98, "⚠ Contains NaN/Inf!", transform=ax.transAxes, 
                   fontsize=10, color="red", verticalalignment="top", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))
        elif np.abs(y).max() > 100:
            ax.text(0.02, 0.98, f"⚠ Max |y| = {np.abs(y).max():.1f}", transform=ax.transAxes,
                   fontsize=10, color="red", verticalalignment="top", bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7))

    # Summary at top
    status = "✓ Healthy"
    if diagnosis["is_exploding"]:
        status = "⚠ Exploding (NaN/Inf detected)"
    elif diagnosis["is_saturated"]:
        status = "⚠ Saturated (output magnitude > 100)"
    elif diagnosis["max_val"] > 50 or diagnosis["min_val"] < -50:
        status = "⚠ Large magnitude (possible instability)"

    fig.suptitle(
        f"KAN Basis Functions — {status}\n"
        f"Max={diagnosis['max_val']:.2f}, Min={diagnosis['min_val']:.2f}, "
        f"Mean|y|={diagnosis['mean_magnitude']:.2f}",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")
    print(f"Diagnosis: {status}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="KANResNet", choices=["KANResNet", "SpatialKANBioheat"])
    ap.add_argument(
        "--checkpoint",
        default="checkpoints/loso/KANResNet/fold_US_001_30W_10min.pth",
    )
    ap.add_argument("--output", default="paper/figures/kan_basis_functions.png")
    args = ap.parse_args()

    # Load model
    frame_shape = (64, 64, 5)
    time_steps = 5

    if args.model == "KANResNet":
        from models.kan import KANResNet

        model = KANResNet(frame_shape=frame_shape)
    elif args.model == "SpatialKANBioheat":
        from models.kan import SpatialKANBioheat

        model = SpatialKANBioheat(frame_shape=frame_shape, time_steps=time_steps, output_hw=(4, 4))
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Load checkpoint
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt)
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}, using untrained model")

    model.eval()

    # Visualize
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_basis_functions(model, args.output)


if __name__ == "__main__":
    main()

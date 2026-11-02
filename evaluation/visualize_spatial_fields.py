"""
Visualize spatial temperature field predictions from spatial models.
Loads checkpoints and generates example predictions for paper.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import DataLoader

from models.backbones import SpatialResNet
from models.kan import SpatialKANBioheat
from physics.models import SpatialPhysicsCNNLSTM
from utils.sequence_dataset import SequenceHeatmapDataset


def _last_field_batch(fields, sequence_length):
    if fields.ndim == 5:
        fields = fields[:, -1]
        if fields.shape[1] == 1:
            fields = fields[:, 0]
    elif fields.ndim == 4:
        if fields.shape[1] == sequence_length:
            fields = fields[:, -1]
        elif fields.shape[1] == 1:
            fields = fields[:, 0]
    if fields.ndim != 3:
        raise ValueError(f"Expected a batch of 2D fields, got shape {fields.shape}.")
    return fields


def load_spatial_model(model_type, checkpoint_path, frame_shape=(64, 64, 5)):
    """Load a spatial model from checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "SpatialResNet":
        model = SpatialResNet(frame_shape=frame_shape)
    elif model_type == "ConvectionBioheat":
        model = SpatialPhysicsCNNLSTM(
            frame_shape=frame_shape, time_steps=5, pretrained=True
        )
    elif model_type == "SpatialKANBioheat":
        model = SpatialKANBioheat(
            frame_shape=frame_shape, time_steps=5, output_hw=(4, 4)
        )
    else:
        raise ValueError(f"Unknown spatial model type: {model_type}")

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"✓ Loaded {model_type} from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}, using untrained model")

    model.to(device)
    model.eval()
    return model, device


def get_sample_batch(batch_size=2):
    """Load a small batch from dataset."""
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        sequence_length=5,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    # Returns: (frames, masks, heatmaps, temps)
    frames, masks, heatmaps, temps = batch
    return frames, heatmaps


def visualize_spatial_predictions():
    """Generate spatial field visualizations for three spatial models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load sample batch
    print("Loading sample batch...")
    frames, heatmaps = get_sample_batch(batch_size=2)
    frames = frames.to(device)
    heatmaps = heatmaps.to(device)

    spatial_models = [
        ("SpatialResNet", "checkpoints/loso/SpatialResNet/fold_US_001.pth"),
        ("ConvectionBioheat", "checkpoints/loso/ConvectionBioheat/fold_US_001.pth"),
        ("SpatialKANBioheat", "checkpoints/loso/SpatialKANBioheat/fold_US_001.pth"),
    ]

    # Each sample gets a prediction and its sparse sensor-measurement target.
    fig, axes = plt.subplots(3, 4, figsize=(14, 12))
    fig.suptitle(
        "Spatial Temperature Field Predictions (Example)",
        fontsize=14,
        fontweight="bold",
    )

    for row_idx, (model_type, checkpoint_path) in enumerate(spatial_models):
        model, _ = load_spatial_model(model_type, checkpoint_path)

        with torch.no_grad():
            # Get predictions
            output = model(frames)

            # Handle different output types/shapes
            if isinstance(output, tuple):
                # Some models return (pred, something_else)
                pred_fields = output[0]
            else:
                pred_fields = output

            pred_field = _last_field_batch(pred_fields, frames.shape[1])
            gt_field = _last_field_batch(heatmaps, frames.shape[1])

            if pred_field.shape[-2:] != gt_field.shape[-2:]:
                pred_field = torch.nn.functional.interpolate(
                    pred_field.unsqueeze(1),
                    size=gt_field.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            pred_field = pred_field.cpu().numpy()
            gt_field = gt_field.cpu().numpy()

        for sample_idx in range(min(2, pred_field.shape[0])):
            pred = pred_field[sample_idx]
            gt = gt_field[sample_idx]
            gt = np.ma.masked_where(gt == 0, gt)
            valid_gt = gt.compressed()
            color_values = [pred.ravel()]
            if valid_gt.size:
                color_values.append(valid_gt)
            vmin = min(values.min() for values in color_values)
            vmax = max(values.max() for values in color_values)
            if vmin == vmax:
                vmax = vmin + 1

            for offset, (field, title) in enumerate(
                ((pred, "Prediction"), (gt, "Sensor ground truth"))
            ):
                ax = axes[row_idx, sample_idx * 2 + offset]
                im = ax.imshow(field, cmap="hot", vmin=vmin, vmax=vmax)
                ax.set_title(f"{model_type} - {title} {sample_idx + 1}", fontsize=9)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = "paper/figures/spatial_field_predictions.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved spatial field predictions to {output_path}")
    plt.close()


def visualize_model_ensemble():
    """Create a comparison panel showing predictions from multiple models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load sample
    frames, heatmaps = get_sample_batch(batch_size=1)
    frames = frames.to(device)

    models_config = {
        "SpatialResNet": "checkpoints/loso/SpatialResNet/fold_US_001.pth",
        "ConvectionBioheat": "checkpoints/loso/ConvectionBioheat/fold_US_001.pth",
        "SpatialKANBioheat": "checkpoints/loso/SpatialKANBioheat/fold_US_001.pth",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        "Spatial Model Predictions: Ground Truth vs. Three Methods",
        fontsize=14,
        fontweight="bold",
    )

    # Ground truth
    gt = heatmaps[0].cpu().numpy()
    # Handle different shapes
    if gt.ndim == 3:  # (T, H, W)
        gt = gt[-1]  # Take last timestep
    elif gt.ndim == 4:  # (T, 1, H, W)
        gt = gt[-1, 0]

    ax = axes[0, 0]
    im = ax.imshow(gt, cmap="hot")
    ax.set_title("Ground Truth", fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, label="Temp. (K)")

    # Model predictions
    for (model_name, checkpoint_path), ax in zip(models_config.items(), axes.flat[1:]):
        model, _ = load_spatial_model(model_name, checkpoint_path)

        with torch.no_grad():
            output = model(frames)

            # Handle tuple output
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output

            # Extract to numpy
            if pred.dim() == 4:
                pred = pred[0, -1].cpu().numpy()
            elif pred.dim() == 3:
                pred = pred[0].cpu().numpy()
            else:
                pred = pred.squeeze().cpu().numpy()

        im = ax.imshow(pred, cmap="hot")
        ax.set_title(model_name, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, label="Pred. (K)")

    plt.tight_layout()
    output_path = "paper/figures/spatial_models_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved spatial models comparison to {output_path}")
    plt.close()


if __name__ == "__main__":
    try:
        visualize_spatial_predictions()
        print("\n" + "=" * 60)
        visualize_model_ensemble()
        print("\n✓ All spatial field visualizations generated successfully!")
    except Exception as e:
        print(f"⚠ Error during visualization: {e}")
        print("  (Checkpoints may not exist for all folds)")

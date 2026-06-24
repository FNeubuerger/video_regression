"""
Visualize spatial temperature field predictions from spatial models.
Loads checkpoints and generates example predictions for paper.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import SpatialResNet
from models.kan import SpatialKANBioheat
from physics.models import SpatialPhysicsCNNLSTM
from utils.sequence_dataset import SequenceHeatmapDataset
from torch.utils.data import DataLoader


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
        use_artifact_masking=True
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
    
    # Create figure with 3 rows (one per model) and 2 columns (two samples)
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle("Spatial Temperature Field Predictions (Example)", fontsize=14, fontweight='bold')
    
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
            
            # Extract last timestep if 4D, otherwise use as-is
            if pred_fields.dim() == 4:  # (B, T, H, W)
                pred_field = pred_fields[:, -1, :, :].cpu().numpy()
            elif pred_fields.dim() == 3:  # (B, H, W)
                pred_field = pred_fields.cpu().numpy()
            else:
                pred_field = pred_fields.squeeze().cpu().numpy()
        
        # Ground truth
        gt_field = heatmaps.cpu().numpy()[:, -1, :, :] if heatmaps.dim() == 4 else heatmaps.cpu().numpy()
        
        # Plot predictions vs ground truth
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            
            # Show prediction
            pred = pred_field[col_idx] if pred_field.ndim == 3 else pred_field
            
            # Normalize to [0, 1] for visualization
            pred_min, pred_max = pred.min(), pred.max()
            if pred_max > pred_min:
                pred_norm = (pred - pred_min) / (pred_max - pred_min)
            else:
                pred_norm = np.zeros_like(pred)
            
            im = ax.imshow(pred_norm, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f"{model_type} - Sample {col_idx+1}", fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Norm. Temp.', fontsize=8)
    
    plt.tight_layout()
    output_path = "paper/figures/spatial_field_predictions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    fig.suptitle("Spatial Model Predictions: Ground Truth vs. Three Methods", fontsize=14, fontweight='bold')
    
    # Ground truth
    gt = heatmaps[0].cpu().numpy()
    # Handle different shapes
    if gt.ndim == 3:  # (T, H, W)
        gt = gt[-1]  # Take last timestep
    elif gt.ndim == 4:  # (T, 1, H, W)
        gt = gt[-1, 0]
    
    ax = axes[0, 0]
    im = ax.imshow(gt, cmap='hot')
    ax.set_title("Ground Truth", fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Temp. (K)')
    
    # Model predictions
    idx = 1
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
        
        im = ax.imshow(pred, cmap='hot')
        ax.set_title(model_name, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Pred. (K)')
    
    plt.tight_layout()
    output_path = "paper/figures/spatial_models_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved spatial models comparison to {output_path}")
    plt.close()


if __name__ == "__main__":
    try:
        visualize_spatial_predictions()
        print("\n" + "="*60)
        visualize_model_ensemble()
        print("\n✓ All spatial field visualizations generated successfully!")
    except Exception as e:
        print(f"⚠ Error during visualization: {e}")
        print("  (Checkpoints may not exist for all folds)")

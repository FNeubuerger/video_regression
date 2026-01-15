"""
Scientific Plotting Script for Temperature Regression Benchmarks

This script generates publication-quality plots to visualize:
1. Temporal Fit (Prediction vs Ground Truth over time)
2. Generalization Performance (Error across different sequences)
3. Spatial Prediction Accuracy (Predicted Heatmaps)
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import sys
from PIL import Image
from torchvision import transforms

# Set publication style
sns.set_theme(style="whitegrid", font="serif")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet, SpatialResNet
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from utils.dataset import TemperatureSequenceDataset

def load_model(model_name, checkpoint_path, device):
    """Load a trained model for plotting."""
    frame_shape = (64, 64, 5)
    time_steps = 5
    
    if "ConvectionBioheat" in model_name or "convection_bioheat" in model_name:
        model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif "BioheatPINN" in model_name or "advanced_bioheat" in model_name:
        model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif "CNNLSTM" in model_name:
        model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif "SimpleResNet" in model_name:
        model = SimpleResNet(frame_shape=frame_shape)
    else:
        # Fallback to SpatialResNet if name matches spatial patterns
        model = SpatialResNet(frame_shape=frame_shape)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Handle DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def plot_temporal_fit(model, model_name, sequence_dir, device, output_path):
    """Plot prediction vs ground truth for a single continuous sequence."""
    print(f"Generating temporal fit plot for {sequence_dir}...")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset for just this sequence
    ds = TemperatureSequenceDataset(
        data_dir="data",
        sequence_length=5,
        transform=transform,
        use_optical_flow=True,
        image_size=(64, 64)
    )
    
    # Filter sequences to only include files from sequence_dir
    # ds.sequences is a list of (image_paths, temperatures)
    seq_name = os.path.basename(sequence_dir)
    filtered_indices = [i for i, seq in enumerate(ds.sequences) if seq_name in seq[0][0]]
    
    if not filtered_indices:
        print(f"No sequences found for {seq_name}")
        return

    all_preds = []
    all_gt = []
    time_indices = []

    with torch.no_grad():
        for i in tqdm(filtered_indices, desc="Sequentially Predicting"):
            images, gt = ds[i]
            images = images.to(device).unsqueeze(0) # (1, T, C, H, W)
            
            out = model(images)
            if out.dim() == 4: # (1, T, 4, 4) or (1, 4, 4)
                 if out.dim() == 4: out = out[:, -1]
                 out = out.mean() # Scalar prediction
            elif out.dim() == 2: # (1, 1) or (1, T)
                 out = out[:, -1].mean()
            
            all_preds.append(out.item())
            all_gt.append(gt.item())
            time_indices.append(i)

    plt.figure(figsize=(12, 6))
    plt.plot(all_gt, label="Ground Truth", color="black", linestyle="--", alpha=0.7)
    plt.plot(all_preds, label=f"Prediction ({model_name})", color="#1f77b4", linewidth=2)
    
    plt.fill_between(range(len(all_gt)), all_gt, all_preds, color="#1f77b4", alpha=0.2, label="Absolute Error")
    
    plt.xlabel("Frame Index (Time)")
    plt.ylabel("Temperature $T/K$")
    plt.title(f"Temporal Fit: Model Prediction vs. Ground Truth ({seq_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved temporal fit to {output_path}")

def plot_generalization_performance(results_csv, output_path):
    """Generate boxplots comparing models across different categories."""
    if not os.path.exists(results_csv):
        print(f"Results CSV {results_csv} not found.")
        return
        
    df = pd.read_csv(results_csv)
    # Remove NaN values for plotting
    df = df.dropna(subset=['RMSE'])
    
    # Filter out "Other" category or "broken" models if needed
    df = df[df['RMSE'] < 10.0]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Display Name', y='RMSE', hue='Category', palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("RMSE $T/K$")
    plt.xlabel("Model Architecture")
    plt.title("Model Performance: Deep Learning vs. Physics-Informed")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved generalization plot to {output_path}")

def main():
    os.makedirs("results/plots", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Temporal Fit Comparison
    best_model_path = "models/convection_bioheat_model.pth"
    if os.path.exists(best_model_path):
        model = load_model("ConvectionBioheat", best_model_path, device)
        # Use sequence_8 as the "unseen" test sequence for the plot
        plot_temporal_fit(model, "Convection Bioheat", "data/sequence_8", device, "results/plots/temporal_fit_physics.png")
    else:
        print("Best model not found for temporal plot.")

    # 2. Generalization Summary
    plot_generalization_performance("results/tables/comprehensive_results.csv", "results/plots/model_comparison_bar.png")

if __name__ == "__main__":
    main()

import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import SimpleResNet
from models.bayesian import BayesianResNet
from utils.sequence_dataset import SequenceHeatmapDataset

def evaluate_ensemble(data_loader, num_models=5, device='cuda'):
    print("Evaluating Ensemble...")
    models = []
    # frame_shape needs to be compatible with what was trained
    # Assuming (64, 64, 5) if using optical flow, or (64, 64, 3)
    # We will try to determine from checkpoint or trycatch
    try:
        model = SimpleResNet(frame_shape=(64, 64, 5)).to(device)
        model.load_state_dict(torch.load(f"checkpoints/ensemble/model_0.pth", map_location=device))
        input_channels = 5
    except:
        input_channels = 3
    
    for i in range(num_models):
        path = f"checkpoints/ensemble/model_{i}.pth"
        if os.path.exists(path):
            model = SimpleResNet(frame_shape=(64, 64, input_channels)).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            models.append(model)
        else:
            print(f"Warning: {path} not found.")

    if not models:
        return np.array([]), np.array([]), np.array([])
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            scalars = None
            if len(batch) == 4:
                images, _, _, scalars = batch
            elif len(batch) == 3:
                images, _, _ = batch
            else:
                images, _ = batch

            if input_channels == 3 and images.shape[2] == 5:
                # If model expects 3 channels but we have 5
                images = images[:, :, :3] # (B, T, 3, H, W)

            # SequenceHeatmapDataset returns (B, T, C, H, W)
            # SimpleResNet might expect (B, C, H, W) or (B, T, C, H, W)
            # If standard SimpleResNet, likely (B, C, H, W)
            # We take last frame? Or reshape (B*T)? 
            # Usually we evaluate last frame.
            if images.dim() == 5:
                images = images[:, -1] # (B, C, H, W)
            
            images = images.to(device)
            
            # Targets
            if scalars is not None:
                # scalars is (B, T, 4)
                targets = scalars[:, -1, :] # (B, 4)
            else:
                 # Fallback?
                 targets = torch.zeros(images.shape[0], 4)

            batch_preds = []
            for model in models:
                out = model(images)
                batch_preds.append(out.cpu().numpy()) # (B, 4)
            
            all_preds.append(np.stack(batch_preds, axis=0)) # Shape: (num_models, B, 4)
            all_targets.append(targets.numpy()) # (B, 4)
            
    # Concatenate batches
    # shapes: list of (num_models, B, 4) -> (num_models, Total, 4)
    all_preds = np.concatenate(all_preds, axis=1) 
    all_targets = np.concatenate(all_targets, axis=0) # (Total, 4)
    
    # Flatten for plotting: (Total*4)
    all_preds = np.concatenate(all_preds, axis=1) # (num_models, Total, 4)
    all_targets = np.concatenate(all_targets, axis=0) # (Total, 4)
    
    # Calculate per-sensor statistics
    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    
    return mean_preds, std_preds, all_targets

def evaluate_bayesian(data_loader, num_samples=10, device='cuda'):
    print("Evaluating Bayesian Model...")
    # Determine input channels from checkpoint ideally, or assume 5
    input_channels = 5
    model = BayesianResNet(frame_shape=(64, 64, input_channels)).to(device)
    
    # Try loading
    path = "models/bayesian_pinn.pth"
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return np.array([]), np.array([]), np.array([])
        
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            scalars = None
            if len(batch) == 4:
                images, _, _, scalars = batch
            else:
                images = batch[0]
            
            if images.dim() == 5:
                images = images[:, -1] # (B, C, H, W)
                
            images = images.to(device)
            
            if scalars is not None:
                targets = scalars[:, -1, :]
            else:
                targets = torch.zeros(images.shape[0], 4)

            batch_preds = []
            # Monte Carlo sampling
            for _ in range(num_samples):
                out = model(images)
                if isinstance(out, tuple): out = out[0]
                batch_preds.append(out.cpu().numpy())
            
            all_preds.append(np.stack(batch_preds, axis=0)) # (Samples, B, 4)
            all_targets.append(targets.numpy())
            
    all_preds = np.concatenate(all_preds, axis=1) # (Samples, Total, 4)
    all_targets = np.concatenate(all_targets, axis=0) # (Total, 4)
    
    all_preds = np.concatenate(all_preds, axis=1) # (Samples, Total, 4)
    all_targets = np.concatenate(all_targets, axis=0) # (Total, 4)
    
    # Calculate per-sensor statistics
    # mean over samples axis=0 -> (Total, 4)
    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    
    return mean_preds, std_preds, all_targets

def plot_uncertainty(targets, means, stds, title, save_path):
    # Determine if per-sensor or flattened
    if targets.ndim > 1 and targets.shape[1] == 4:
        # Create 4 subplots for sensors
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i in range(4):
            ax = axes[i]
            t = targets[:, i]
            m = means[:, i]
            s = stds[:, i]
            
            # Sort
            sorted_idx = np.argsort(t)
            t_sorted = t[sorted_idx]
            m_sorted = m[sorted_idx]
            s_sorted = s[sorted_idx]
            
            # Subset
            if len(t) > 100:
                step = len(t) // 100
                idx = np.arange(0, len(t), step)
                t_plot = t_sorted[idx]
                m_plot = m_sorted[idx]
                s_plot = s_sorted[idx]
            else:
                t_plot, m_plot, s_plot = t_sorted, m_sorted, s_sorted
                
            ax.plot(t_plot, label='Ground Truth', color='black', linewidth=2)
            ax.errorbar(range(len(t_plot)), m_plot, yerr=s_plot, fmt='o', alpha=0.5, label='Pred $\pm$ Std')
            ax.set_title(f"Sensor {i}")
            ax.set_ylabel("Temp (K)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Also compute and print aggregate metrics per sensor
        print(f"\n--- {title} Per-Sensor Metrics ---")
        for i in range(4):
            mse = np.mean((targets[:, i] - means[:, i])**2)
            mae = np.mean(np.abs(targets[:, i] - means[:, i]))
            unc_cal = np.mean(stds[:, i])
            print(f"Sensor {i}: RMSE={np.sqrt(mse):.3f}, MAE={mae:.3f}, Mean Std={unc_cal:.3f}")
            
    else:
        # Fallback to flattened plot
        plt.figure(figsize=(12, 6))
        indices = np.arange(len(targets))
        
        # Sort by target temperature for better visualization
        sorted_indices = np.argsort(targets)
        targets = targets[sorted_indices]
        means = means[sorted_indices]
        stds = stds[sorted_indices]
        
        # Plot only a subset for clarity if too many points
        if len(targets) > 100:
            step = len(targets) // 100
            indices = np.arange(0, len(targets), step)
            targets = targets[indices]
            means = means[indices]
            stds = stds[indices]
            
        plt.plot(targets, label='Ground Truth', color='black', linewidth=2)
        plt.errorbar(range(len(targets)), means, yerr=stds, fmt='o', alpha=0.5, label='Prediction $\pm$ Std Dev')
        
        plt.title(title)
        plt.xlabel('Sample Index (Sorted by Temperature)')
        plt.ylabel('Temperature $T/K$')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300)
        plt.close()
    print(f"Plot saved to {save_path}")
    plt.ylabel('Temperature $T/K$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Use SequenceHeatmapDataset for new data (level1_cropped)
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        use_optical_flow=True,
        sequence_length=5
    )
    
    # Use a small subset for quick evaluation
    subset_indices = np.random.choice(len(dataset), min(200, len(dataset)), replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    os.makedirs("results/uncertainty", exist_ok=True)
    
    # Evaluate Ensemble
    if os.path.exists("checkpoints/ensemble"):
        ens_mean, ens_std, ens_targets = evaluate_ensemble(loader, device=device)
        if ens_mean.size > 0:
            plot_uncertainty(ens_targets, ens_mean, ens_std, 
                            "Ensemble Uncertainty Estimation", 
                            "results/uncertainty/ensemble_plot.png")
    
    # Evaluate Bayesian
    # Uses models/bayesian_pinn.pth as defined in evaluate_bayesian
    bay_mean, bay_std, bay_targets = evaluate_bayesian(loader, device=device)
    if bay_mean.size > 0:
        plot_uncertainty(bay_targets, bay_mean, bay_std, 
                        "Bayesian Uncertainty Estimation", 
                        "results/uncertainty/bayesian_plot.png")

if __name__ == "__main__":
    main()

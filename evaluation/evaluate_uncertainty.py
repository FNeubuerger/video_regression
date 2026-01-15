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
from utils.heatmap_dataset import TemperatureHeatmapDataset

def evaluate_ensemble(data_loader, num_models=5, device='cuda'):
    print("Evaluating Ensemble...")
    models = []
    for i in range(num_models):
        model = SimpleResNet(frame_shape=(64, 64, 3)).to(device)
        model.load_state_dict(torch.load(f"checkpoints/ensemble/model_{i}.pth", map_location=device))
        model.eval()
        models.append(model)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            # Handle variable unpacking from dataset (img, target) or (img, target, mask)
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch
                
            images = images.to(device)
            # Flatten dense labels if model predicts scalar? 
            # SimpleResNet predicts scalar (B, 1). Labels are (B, 1, 64, 64)
            # Take max of labels
            if labels.dim() > 2:
                labels = labels.amax(dim=(1, 2, 3)).unsqueeze(1) # (B, 1)

            batch_preds = []
            for model in models:
                batch_preds.append(model(images).cpu().numpy())
            
            all_preds.append(np.stack(batch_preds, axis=0)) # Shape: (num_models, batch_size)
            all_targets.append(labels.numpy())
            
    all_preds = np.concatenate(all_preds, axis=1) # Shape: (num_models, total_samples)
    all_targets = np.concatenate(all_targets)
    
    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    
    return mean_preds, std_preds, all_targets

def evaluate_bayesian(data_loader, num_samples=10, device='cuda'):
    print("Evaluating Bayesian Model...")
    model = BayesianResNet(frame_shape=(64, 64, 3)).to(device)
    model.load_state_dict(torch.load("checkpoints/bayesian_resnet.pth", map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            batch_preds = []
            # Monte Carlo sampling
            for _ in range(num_samples):
                batch_preds.append(model(images).cpu().numpy())
            
            all_preds.append(np.stack(batch_preds, axis=0))
            all_targets.append(labels.numpy())
            
    all_preds = np.concatenate(all_preds, axis=1)
    all_targets = np.concatenate(all_targets)
    
    mean_preds = np.mean(all_preds, axis=0)
    std_preds = np.std(all_preds, axis=0)
    
    return mean_preds, std_preds, all_targets

def plot_uncertainty(targets, means, stds, title, save_path):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Use TemperatureHeatmapDataset for new data (level1_cropped)
    # It returns (image, dense_target, mask) or similar.
    # We need to adapt the unpacking in the eval loops or wrapper.
    # But wait, evaluate_ensemble loops: for images, labels in data_loader
    # TemperatureHeatmapDataset returns (img, target, mask) if mask=True
    # Default mask=True in some places, mask=False in others.
    # __init__(..., use_artifact_masking=False) default.
    # __getitem__ returns (img, target, mask) ALWAYS? 
    # Let's check TemperatureHeatmapDataset.__getitem__ return signature.
    # If it returns 3 values, DataLoader yields list of 3.
    # The loop `for images, labels in` will fail.
    
    # Check Heatmap Dataset __getitem__
    # If I cannot verify return signature, I assume it returns 3 items based on other files.
    # I will stick to what works.
    
    dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        target_size=(64, 64),
        raw_dir="data/level0_raw",
        transform=transform
    )
    
    # Use a small subset for quick evaluation
    subset_indices = np.random.choice(len(dataset), 200, replace=False)
    subset = torch.utils.data.Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    os.makedirs("results/uncertainty", exist_ok=True)
    
    # Evaluate Ensemble
    if os.path.exists("checkpoints/ensemble/model_0.pth"):
        ens_mean, ens_std, ens_targets = evaluate_ensemble(loader, device=device)
        plot_uncertainty(ens_targets, ens_mean, ens_std, 
                        "Ensemble Uncertainty Estimation", 
                        "results/uncertainty/ensemble_plot.png")
    
    # Evaluate Bayesian
    if os.path.exists("checkpoints/bayesian_resnet.pth"):
        bay_mean, bay_std, bay_targets = evaluate_bayesian(loader, device=device)
        plot_uncertainty(bay_targets, bay_mean, bay_std, 
                        "Bayesian Uncertainty Estimation", 
                        "results/uncertainty/bayesian_plot.png")

if __name__ == "__main__":
    main()

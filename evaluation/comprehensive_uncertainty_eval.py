import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureRegressionDataset
from utils.model_registry import MODEL_REGISTRY

def compute_uncertainty_metrics(targets, means, stds):
    """
    Computes uncertainty quantification metrics.
    
    Args:
        targets (np.array): Ground truth values (N,)
        means (np.array): Predictive means (N,)
        stds (np.array): Predictive standard deviations (N,)
        
    Returns:
        dict: Dictionary of metrics
    """
    # 1. Standard Regression Metrics
    mse = np.mean((targets - means)**2)
    mae = np.mean(np.abs(targets - means))
    rmse = np.sqrt(mse)
    
    # 2. Negative Log Likelihood (Gaussian Assumption)
    # NLL = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
    nll = 0.5 * np.log(2 * np.pi * stds**2) + (targets - means)**2 / (2 * stds**2)
    mean_nll = np.mean(nll)
    
    # 3. Prediction Interval Coverage Probability (PICP)
    # Percentage of targets falling within 95% CI (mean +/- 1.96 * std)
    ci_lower = means - 1.96 * stds
    ci_upper = means + 1.96 * stds
    in_interval = (targets >= ci_lower) & (targets <= ci_upper)
    picp = np.mean(in_interval)
    
    # 4. Mean Prediction Interval Width (MPIW)
    mpiw = np.mean(ci_upper - ci_lower)
    
    return {
        "MSE": float(mse),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "NLL": float(mean_nll),
        "PICP_95": float(picp),
        "MPIW_95": float(mpiw)
    }

def evaluate_model(model_name, checkpoint_path, data_dir, batch_size=32, num_samples=50, device='cuda', limit=None):
    print(f"Evaluating {model_name} from {checkpoint_path}")
    
    # 1. Load Model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    model = ModelClass(**kwargs)
    
    # Load weights
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 2. Load Data
    # Use 'test' split if available, otherwise use a subset of data
    # Assuming data_dir has sequence folders. We'll use the standard dataset class.
    # Ideally we should have a separate test set. For now, we'll use the dataset class.
    dataset = TemperatureRegressionDataset(data_dir=data_dir, sequence_length=5)
    
    # Create a simple train/test split (80/20) for demonstration if no explicit test set
    # In a real scenario, we would load a specific test set.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    
    if limit is not None and limit < len(test_dataset):
        print(f"Limiting evaluation to {limit} samples.")
        indices = torch.randperm(len(test_dataset))[:limit]
        test_dataset = torch.utils.data.Subset(test_dataset, indices)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Test set size: {len(test_dataset)} samples")
    
    # 3. Inference Loop
    all_means = []
    all_stds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Monte Carlo Sampling
            batch_preds = []
            for _ in range(num_samples):
                # Forward pass
                # Note: Some models might return multiple outputs (e.g. physics loss components)
                # We assume the first output is the prediction
                out = model(images)
                if isinstance(out, tuple):
                    out = out[0]
                
                # Handle sequence output (B, T) -> (B,)
                # If the model outputs a sequence of predictions, we take the last one
                # to match the 'last' strategy of the dataset.
                if out.dim() > 1 and out.shape[1] > 1:
                     out = out[:, -1]
                     
                batch_preds.append(out.cpu().numpy())
            
            # Stack predictions: (num_samples, batch_size)
            batch_preds = np.stack(batch_preds, axis=0)
            
            # Compute statistics
            batch_mean = np.mean(batch_preds, axis=0)
            batch_std = np.std(batch_preds, axis=0)
            
            all_means.append(batch_mean)
            all_stds.append(batch_std)
            all_targets.append(targets.numpy())
            
    # Concatenate all batches
    means = np.concatenate(all_means)
    stds = np.concatenate(all_stds)
    targets = np.concatenate(all_targets)
    
    # 4. Compute Metrics
    metrics = compute_uncertainty_metrics(targets, means, stds)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    return metrics, targets, means, stds

def plot_results(targets, means, stds, model_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Calibration Plot (Predicted vs Actual)
    plt.figure(figsize=(10, 6))
    plt.errorbar(targets, means, yerr=stds, fmt='o', alpha=0.2, label='Predictions')
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title(f'{model_name}: Predicted vs Actual with Uncertainty')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_calibration.png'))
    plt.close()
    
    # 2. Uncertainty vs Error
    errors = np.abs(targets - means)
    plt.figure(figsize=(10, 6))
    plt.scatter(stds, errors, alpha=0.3)
    plt.xlabel('Predicted Uncertainty (Std Dev)')
    plt.ylabel('Absolute Error')
    plt.title(f'{model_name}: Uncertainty vs Error')
    
    # Add correlation coefficient
    corr = np.corrcoef(stds, errors)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes)
    
    plt.savefig(os.path.join(output_dir, f'{model_name}_uncertainty_vs_error.png'))
    plt.close()
    
    # 3. Sorted Prediction Interval Plot
    plt.figure(figsize=(12, 6))
    # Sort by target value
    sorted_indices = np.argsort(targets)
    # Downsample for visibility if needed
    if len(targets) > 200:
        step = len(targets) // 200
        indices = sorted_indices[::step]
    else:
        indices = sorted_indices
        
    plt.plot(targets[indices], 'k-', label='Ground Truth')
    plt.fill_between(range(len(indices)), 
                     means[indices] - 1.96 * stds[indices], 
                     means[indices] + 1.96 * stds[indices], 
                     color='blue', alpha=0.3, label='95% CI')
    plt.plot(means[indices], 'b--', alpha=0.6, label='Mean Prediction')
    
    plt.title(f'{model_name}: Predictions with 95% Confidence Intervals (Sorted)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_prediction_intervals.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Bayesian Models with Uncertainty Metrics")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., BayesianResNet)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results/uncertainty_eval", help="Directory to save results")
    parser.add_argument("--samples", type=int, default=50, help="Number of Monte Carlo samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples for quick evaluation")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    metrics, targets, means, stds = evaluate_model(
        args.model, 
        args.checkpoint, 
        args.data_dir, 
        batch_size=args.batch_size, 
        num_samples=args.samples,
        device=device,
        limit=args.limit
    )
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{args.model}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate Plots
    plot_results(targets, means, stds, args.model, args.output_dir)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()

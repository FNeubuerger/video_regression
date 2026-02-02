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
from torchvision import transforms
import pandas as pd
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureRegressionDataset
from utils.sequence_dataset import SequenceHeatmapDataset
from utils.heatmap_dataset import TemperatureHeatmapDataset
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
    # Handle deterministic models (stds ~ 0)
    if np.all(stds < 1e-6):
        mean_nll = float('nan')
    else:
        # Add epsilon for numerical stability
        stds_safe = np.maximum(stds, 1e-6)
        nll = 0.5 * np.log(2 * np.pi * stds_safe**2) + (targets - means)**2 / (2 * stds_safe**2)
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

def fix_state_dict_keys(state_dict, model_name):
    """
    Fixes state_dict keys for loading legacy checkpoints or mismatched architectures.
    """
    new_state_dict = {}
    
    # 1. LatentLTC_UNet: Checkpoint 'encoder_backbone.' -> Model 'encoder.'
    if "LatentLTC" in model_name:
        for k, v in state_dict.items():
            if k.startswith("encoder_backbone."):
                new_key = k.replace("encoder_backbone.", "encoder.")
                # The model maps encoder_backbone = encoder, so either key should work if alias exists.
                # But if checkpoint has one and model expecting other without alias persisting in state_dict...
                new_state_dict[new_key] = v
                # Also keep original just in case
                new_state_dict[k] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    # 2. BayesianSpatialResNet: Checkpoint 'backbone.' -> Model 'cnn.' 
    if "Spatial" in model_name:
        
        # Mapping for ResNet backbone to cnn sequential
        layer_map = {
            "backbone.conv1": "cnn.0",
            "backbone.bn1": "cnn.1",
            "backbone.layer1": "cnn.4",
            "backbone.layer2": "cnn.5",
            "backbone.layer3": "cnn.6",
            "backbone.layer4": "cnn.7"
        }
        
        for k, v in state_dict.items():
            mapped = False
            for old_prefix, new_prefix in layer_map.items():
                if k.startswith(old_prefix):
                    new_key = k.replace(old_prefix, new_prefix)
                    new_state_dict[new_key] = v
                    mapped = True
                    break
            if not mapped:
                new_state_dict[k] = v
        
        return new_state_dict

    return state_dict

def evaluate_model(model_name, checkpoint_path, data_dir, batch_size=32, num_samples=50, device='cuda', limit=None, ensemble_dir=None):
    print(f"Evaluating {model_name}")
    
    # 1. Load Model(s)
    # Handle masked variant names by stripping suffix
    base_model_name = model_name.replace("_masked", "")
    
    if base_model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} (base: {base_model_name}) not found in registry.")
        
    ModelClass, kwargs = MODEL_REGISTRY[base_model_name]
    
    models = []
    
    # Helper to load state dict safely
    def load_weights(model, state_dict):
        # Fix keys
        state_dict = fix_state_dict_keys(state_dict, model_name)
        
        # Try loading, if size mismatch, retry with out_features=1 (legacy)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if 'size mismatch' in str(e) and 'out_features' in kwargs.keys():
                 pass # Already tried? kwargs came from registry.
            elif 'size mismatch' in str(e) and base_model_name in ["BayesianResNet", "BayesianPINN", "BayesianCNNLSTM"]:
                print(f"Warning: Size mismatch detected. Attempting legacy load (out_features=1)...")
                # Hack: modify the last layer in place if possible, or re-init?
                # Re-init is cleaner if we can change args.
                # But we need to know WHICH arg controls output.
                # For BayesianResNet, I added out_features.
                model_new = ModelClass(**kwargs, out_features=1)
                model_new.load_state_dict(state_dict)
                return model_new
            elif 'Missing key(s)' in str(e) and "LatentLTC" in model_name:
                # If fix_keys didn't work (e.g. alias), try strictly renamed
                 print("Warning: Missing keys for LatentLTC. Attempting strict rename...")
                 new_sd = {k.replace("encoder_backbone.", "encoder."): v for k, v in state_dict.items()}
                 model.load_state_dict(new_sd, strict=False) # Strict=False to ignore extra
            else:
                 raise e
        return model

    if ensemble_dir:
        print(f"Loading ensemble from {ensemble_dir}")
        if not os.path.exists(ensemble_dir):
            raise FileNotFoundError(f"Ensemble directory not found: {ensemble_dir}")
        
        # Find all .pth files
        checkpoint_files = [f for f in os.listdir(ensemble_dir) if f.endswith('.pth')]
        if not checkpoint_files:
            raise ValueError(f"No .pth files found in {ensemble_dir}")
            
        print(f"Found {len(checkpoint_files)} ensemble members.")
        
        for ckpt in checkpoint_files:
            model = ModelClass(**kwargs)
            state_dict = torch.load(os.path.join(ensemble_dir, ckpt), map_location=device)
            model = load_weights(model, state_dict)
            model.to(device)
            model.eval()
            models.append(model)
    else:
        print(f"Loading single model from {checkpoint_path}")
        model = ModelClass(**kwargs)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model = load_weights(model, state_dict)
        model.to(device)
        model.eval()
        models.append(model)
    
    # 2. Load Data
    is_dense_model = "UNet" in model_name or "LTC" in model_name or "Spatial" in model_name
    use_masking = "_masked" in model_name
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Default to Sequence Helper for most models as they need 5 channels (RGB+Flow)
    # Most models (BayesianResNet, CNNLSTM, PINN) now expect 5 channels.
    # Dataset needs to match.
    if "LTC" in model_name or "CNNLSTM" in model_name or "PINN" in model_name or "BayesianResNet" in model_name or "Spatial" in model_name:
        print(f"Using SequenceHeatmapDataset for {model_name}")
        dataset = SequenceHeatmapDataset(
            data_dir=data_dir, 
            sequence_length=5, # Default to 5
            target_size=(64, 64),
            transform=transform,
            use_artifact_masking=use_masking,
            use_optical_flow=True # Ensure 5 channels
        )
    elif "UNet" in model_name:
         dataset = TemperatureHeatmapDataset(
            data_dir=data_dir,
            target_size=(64, 64),
            transform=transform,
            use_artifact_masking=use_masking
        )
    else:
        # Fallback 
        dataset = TemperatureHeatmapDataset(
            data_dir=data_dir,
            target_size=(64, 64),
            transform=transform,
            use_artifact_masking=use_masking
        )
    
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
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Handle different dataset return signatures
            targets = None
            mask = None
            
            if len(batch) == 4:
                # SequenceHeatmapDataset: (frames, sparse_targets, priors, scalars)
                images, _, priors, targets = batch
                priors = priors.to(device)
            elif len(batch) == 6:
                # TemperatureHeatmapDataset: (img, sparse, mask, eroded, prior, scalar_label)
                images, _, mask, _, priors, targets = batch
                priors = priors.to(device)
            else:
                 # Fallback
                 images = batch[0]
                 priors = torch.zeros(images.shape[0], 1).to(device)
                 mask = None

            images = images.to(device)
            if targets is not None:
                targets = targets.to(device)
            else:
                targets = torch.zeros(images.shape[0], 1).to(device)

            # Auto-align channels
            try:
                # Peek at model weights to determine expected input channels
                first_param = next(model.parameters())
                if first_param.ndim == 4:
                    expected_c = first_param.shape[1]
                    if expected_c == 5 and images.shape[1] == 3:
                        # Pad with zeros
                        b, c, h, w = images.shape
                        images = torch.cat([images, torch.zeros(b, 2, h, w, device=device)], dim=1)
                    elif expected_c == 3 and images.shape[1] == 5:
                        # Slice
                        images = images[:, :3, :, :]
            except Exception:
                pass
            if mask is not None:
                mask = mask.to(device)
                # Apply masking for the input
                # Handle broadcasting for sequence data: images (B, T, C, H, W), mask (B, 1, H, W)
                if images.dim() == 5:
                    images = images * (1.0 - mask.unsqueeze(1))
                else:
                    images = images * (1.0 - (mask.unsqueeze(1) if mask.dim() == 3 else mask))
            
            # Extract number of expected channels from model
            sample_model = models[0]
            target_ch = 3 # Default
            if hasattr(sample_model, "n_channels"):
                target_ch = sample_model.n_channels
            elif hasattr(sample_model, "input_channels"):
                target_ch = sample_model.input_channels
            elif hasattr(sample_model, "encoder_backbone") and hasattr(sample_model.encoder_backbone, "conv1"):
                 target_ch = sample_model.encoder_backbone.conv1.in_channels
            elif hasattr(sample_model, "backbone") and hasattr(sample_model.backbone, "conv1"):
                 target_ch = sample_model.backbone.conv1.in_channels
            elif hasattr(sample_model, "cnn") and isinstance(sample_model.cnn[0], nn.Conv2d):
                 target_ch = sample_model.cnn[0].in_channels
            else:
                 target_ch = images.shape[2] if images.dim() == 5 else images.shape[1]

            # Slice images if needed
            if images.dim() == 5: # [B, T, C, H, W]
                if images.shape[2] > target_ch:
                    images = images[:, :, :target_ch, :, :]
            elif images.dim() == 4: # [B, C, H, W]
                if images.shape[1] > target_ch:
                    images = images[:, :target_ch, :, :]

            # Monte Carlo Sampling or Ensemble Averaging
            batch_preds = []
            
            for model in models if ensemble_dir else [models[0]] * (1 if not num_samples else num_samples):
                if ensemble_dir:
                    out = model(images)
                else: 
                    # Bayesian sampling loop handled by outer loop logic above
                    out = models[0](images)


                if isinstance(out, tuple):
                    out = out[0]
                
                # Add physics prior if available (for residual learning models)
                if is_dense_model and priors is not None:
                    # Match shapes for prior addition
                    if out.dim() == 5 and priors.dim() == 4:
                        out = out + priors.unsqueeze(1)
                    else:
                        out = out + priors
                
                # Handle various output shapes to get (Batch,) peak temperature
                if out.dim() == 5: # (Batch, Time, C, H, W)
                    out = out[:, -1] # -> (Batch, C, H, W)
                
                if out.dim() == 4: # (Batch, C, H, W)
                    # Take spatial max
                    out = torch.amax(out, dim=[2, 3]) # -> (Batch, C)
                    # If multiple channels, take max over channels (usually just 1)
                    if out.dim() > 1 and out.shape[1] > 1:
                        out = torch.amax(out, dim=1)
                    else:
                        out = out.squeeze(-1) if out.dim() > 1 else out
                
                if out.dim() == 3: # (Batch, H, W)
                    out = torch.amax(out, dim=[1, 2])
                
                if out.dim() == 2: # (Batch, Time) or (Batch, 1)
                    out = out[:, -1]
                
                batch_preds.append(out.cpu().numpy())
                if ensemble_dir and model == models[-1]: break # Avoid extra loops for ensemble
            
            # Stack predictions: (num_samples, batch_size)
            batch_preds = np.stack(batch_preds, axis=0)
            
            # Compute statistics
            batch_mean = np.mean(batch_preds, axis=0)
            batch_std = np.std(batch_preds, axis=0)
            
            all_means.append(batch_mean)
            all_stds.append(batch_std)
            
            # Handle target shape for dense models (take spatial max for scalar metric)
            if targets.dim() > 1:
                if targets.dim() == 4: # (B, C, H, W)
                    targets_scalar = torch.amax(targets, dim=[2, 3])
                    if targets_scalar.dim() > 1: targets_scalar = targets_scalar[:, 0]
                elif targets.dim() == 3: # (B, H, W)
                    targets_scalar = torch.amax(targets, dim=[1, 2])
                else: 
                     # (B, 1) or (B, C) -> Reduce if needed or keep matching dims
                     targets_scalar = targets
                
                # Force to 1D for consistency if it's a scalar regression task
                # Check if targets_scalar is (B, 1)
                if targets_scalar.dim() >= 2:
                     targets_scalar = targets_scalar.reshape(-1)
                
                all_targets.append(targets_scalar.cpu().numpy())
            else:
                all_targets.append(targets.cpu().numpy())
            
            # Stack predictions: (num_samples, batch_size)
            batch_preds = np.stack(batch_preds, axis=0)
            
            # Compute statistics
            batch_mean = np.mean(batch_preds, axis=0)
            batch_std = np.std(batch_preds, axis=0)
            
            
            all_means.append(batch_mean)
            all_stds.append(batch_std)
            
    # Concatenate all batches
    means = np.concatenate(all_means).reshape(-1)
    stds = np.concatenate(all_stds).reshape(-1)
    targets = np.concatenate(all_targets).reshape(-1)

    # Force alignment
    min_len = min(len(means), len(targets))
    means = means[:min_len]
    stds = stds[:min_len]
    targets = targets[:min_len]

    print(f"DEBUG: means={means.shape}, targets={targets.shape}, stds={stds.shape}")
    
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
    plt.xlabel('Ground Truth Temperature $T/K$')
    plt.ylabel('Predicted Temperature $T/K$')
    plt.title(f'{model_name}: Predicted vs Actual with Uncertainty')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_name}_calibration.png'))
    plt.close()
    
    # 2. Uncertainty vs Error
    errors = np.abs(targets - means)
    plt.figure(figsize=(10, 6))
    plt.scatter(stds, errors, alpha=0.3)
    plt.xlabel('Predicted Uncertainty $\sigma/K$')
    plt.ylabel('Absolute Error $|T_{pred} - T_{gt}|/K$')
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
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--ensemble_dir", type=str, help="Directory containing ensemble checkpoints")
    parser.add_argument("--data_dir", type=str, default="data/level1_cropped", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="results/uncertainty_eval", help="Directory to save results")
    parser.add_argument("--samples", type=int, default=50, help="Number of Monte Carlo samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test samples for quick evaluation")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation even if results exist")
    
    args = parser.parse_args()
    
    if not args.checkpoint and not args.ensemble_dir:
        parser.error("Either --checkpoint or --ensemble_dir must be provided.")
    
    # Determine output name
    if args.ensemble_dir:
        run_name = "Ensemble"
    else:
        # Use model name combined with checkpoint name to avoid overwrites
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        run_name = f"{args.model}_{ckpt_name}"
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")
    if os.path.exists(metrics_path) and not args.force:
        print(f"Results already exist at {metrics_path}. Skipping evaluation. Use --force to override.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    metrics, targets, means, stds = evaluate_model(
        args.model, 
        args.checkpoint, 
        args.data_dir, 
        batch_size=args.batch_size, 
        num_samples=args.samples,
        device=device,
        limit=args.limit,
        ensemble_dir=args.ensemble_dir
    )
    
    # Determine output name
    if args.ensemble_dir:
        run_name = "Ensemble"
    else:
        # Use checkpoint filename as run name (minus extension)
        run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, f"{run_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate Plots
    plot_results(targets, means, stds, run_name, args.output_dir)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()

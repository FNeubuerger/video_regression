import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset
from utils.model_registry import MODEL_REGISTRY

def calculate_uncertainty_metrics(pred_mean, pred_std, true_map, mask_map):
    """
    Calculates uncertainty metrics: NLL, PICP, MPIW.
    Only evaluates on the masked regions (sensors) or full map if available.
    
    Args:
        pred_mean (np.ndarray): (H, W) Predicted Mean
        pred_std (np.ndarray): (H, W) Predicted Std Dev
        true_map (np.ndarray): (H, W) Ground Truth
        mask_map (np.ndarray): (H, W) Mask
    """
    # Select valid pixels
    valid_mask = mask_map > 0
    if np.sum(valid_mask) == 0:
        return {}
        
    y_true = true_map[valid_mask]
    y_pred = pred_mean[valid_mask]
    y_std = pred_std[valid_mask] + 1e-6 # Avoid div by zero
    
    # 1. NLL (Gaussian)
    # NLL = 0.5 * log(2*pi*sigma^2) + (y - mu)^2 / (2*sigma^2)
    nll = 0.5 * np.log(2 * np.pi * y_std**2) + (y_true - y_pred)**2 / (2 * y_std**2)
    avg_nll = np.mean(nll)
    
    # 2. PICP (Prediction Interval Coverage Probability)
    # 95% CI is mu +/- 1.96 * sigma
    ci_lower = y_pred - 1.96 * y_std
    ci_upper = y_pred + 1.96 * y_std
    within_ci = (y_true >= ci_lower) & (y_true <= ci_upper)
    picp = np.mean(within_ci)
    
    # 3. MPIW (Mean Prediction Interval Width)
    width = ci_upper - ci_lower
    mpiw = np.mean(width)
    
    return {
        'nll': avg_nll,
        'picp': picp,
        'mpiw': mpiw
    }

def calculate_metrics(pred_map, true_map, mask_map, safe_threshold=43.0, ablation_threshold=50.0):
    """
    Calculates dense map metrics.
    
    Args:
        pred_map (np.ndarray): (H, W) Predicted heatmap.
        true_map (np.ndarray): (H, W) Ground truth heatmap (sparse or dense).
        mask_map (np.ndarray): (H, W) Mask indicating where GT is valid.
        
    Returns:
        dict: Metrics.
    """
    metrics = {}
    
    # 1. Pixel-wise Errors (only where mask is active for sparse GT, or full for dense)
    # Check if mask is sparse (only a few pixels) or dense region
    is_sparse = np.sum(mask_map) < (mask_map.size * 0.1) # < 10% coverage
    
    if is_sparse:
        # Sparse Evaluation (Part 1 style but on map)
        diff = (pred_map - true_map) * mask_map
        mse = np.sum(diff**2) / np.sum(mask_map)
        mae = np.sum(np.abs(diff)) / np.sum(mask_map)
        metrics['sparse_rmse'] = np.sqrt(mse)
        metrics['sparse_mae'] = mae
    else:
        # Dense Evaluation (if we had dense ground truth, e.g. simulation)
        # Assuming true_map is dense here
        metrics['dense_rmse'] = np.sqrt(np.mean((pred_map - true_map)**2))
    

    pred_max = np.max(pred_map)
    true_sensors_max = np.max(true_map[mask_map > 0]) if np.any(mask_map > 0) else 0
    metrics['peak_error'] = abs(pred_max - true_sensors_max)
    
    # 2. Safety Violation (Hausdorff Proxy)
    # If we assume the 4 sensors bound the "Hot Zone", we can check if 
    # the predicted 43C contour extends heavily beyond the sensors? hard to define without dense GT.
    
    # 3. Smoothness / TV Norm (Regularization Metric)
    # Measures how "noisy" the prediction is.
    tv = np.sum(np.abs(np.diff(pred_map, axis=0))) + np.sum(np.abs(np.diff(pred_map, axis=1)))
    metrics['tv_norm'] = tv / pred_map.size
    
    return metrics

def evaluate_dense(model_name, checkpoint_path, device='cuda', save_plots=False, num_mc_samples=20):
    print(f"Evaluating {model_name} on Dense Metrics (MC Samples={num_mc_samples})...")
    
    # Load Model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    
    # Attempt to load model with default registry args (usually 5 channels)
    try:
        model = ModelClass(**kwargs)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("Loaded model with default registry configuration.")
        input_channels = kwargs.get('n_channels', 3)
    except Exception as e:
        print(f"Failed to load with registry args ({e}). Trying n_channels=3 compatibility mode...")
        # Fallback for models trained with 3 channel input (legacy/hybrid script)
        if 'n_channels' in kwargs:
            kwargs['n_channels'] = 3
        model = ModelClass(**kwargs)
        model.load_state_dict(checkpoint)
        input_channels = 3
        print("Loaded model with n_channels=3.")

    model.to(device)
    model.eval()
    
    # Determine if Bayesian/Variational
    is_variational = getattr(model, 'variational', False)
    # Ensure dropout is ON if we rely on MC Dropout (Deep Ensembles uses multiple models, BNN uses variational layers)
    # For Variational layers (BayesByBackprop), calling forward() samples weights.
    
    # Load Dataset
    # We use validation data
    dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        use_physics_prior=True
    )
    
    # Evaluate
    results = {
        'sparse_rmse': [],
        'peak_error': [],
        'tv_norm': [],
        'nll': [],
        'picp': [],
        'mpiw': []
    }
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            frame, target_map, mask_map, temps, prior = dataset[i]
            
            # Prepare Input
            # Dataset returns (3, H, W). 
            if input_channels == 5:
                # Pad if we promised 5 channels but only have 3
                # Ideally we load optical flow, but for now we pad
                inp = torch.zeros((1, 5, 64, 64)).to(device)
                inp[0, :3] = frame.to(device)
            else:
                inp = frame.unsqueeze(0).to(device)
                
            # MC Sampling for Uncertainty
            preds_stack = []
            
            # If not variational, we just run once (unless using MC Dropout, but let's assume BNN specific)
            passes = num_mc_samples if is_variational else 1
            
            for _ in range(passes):
                if is_variational:
                    # Expect tuple (pred, kl)
                    out, _ = model(inp)
                else:
                    out = model(inp)
                    
                preds_stack.append(out.squeeze().cpu().numpy())
            
            preds_stack = np.array(preds_stack) # (N, H, W)
            
            pred_mean = np.mean(preds_stack, axis=0) # (H, W)
            pred_std = np.std(preds_stack, axis=0)   # (H, W)
            
            target_map_np = target_map.squeeze().numpy()
            mask_map_np = mask_map.squeeze().numpy()
            
            # Deterministic Metrics (using Mean)
            m = calculate_metrics(pred_mean, target_map_np, mask_map_np)
            for k, v in m.items():
                results[k].append(v)
                
            # Uncertainty Metrics
            if is_variational:
                u_m = calculate_uncertainty_metrics(pred_mean, pred_std, target_map_np, mask_map_np)
                for k, v in u_m.items():
                    results[k].append(v)
            
            if save_plots and i < 5:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 4, 1)
                plt.imshow(pred_mean, cmap='inferno')
                plt.title("Prediction (Mean)")
                plt.colorbar()
                
                plt.subplot(1, 4, 2)
                plt.imshow(pred_std, cmap='viridis')
                plt.title("Uncertainty (Std)")
                plt.colorbar()
                
                plt.subplot(1, 4, 3)
                plt.imshow(target_map_np, cmap='inferno')
                plt.title("Sparse GT")
                plt.colorbar()
                
                plt.subplot(1, 4, 4)
                plt.imshow(prior.squeeze(), cmap='jet')
                plt.title("Physics Prior")
                plt.savefig(f"results/plots/dense_eval_{model_name}_{i}.png")
                plt.close()

    # Aggregate
    print("\nResults:")
    for k, v in results.items():
        if len(v) > 0:
            print(f"{k}: {np.mean(v):.4f}")
        else:
            print(f"{k}: N/A")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--mc_samples", type=int, default=20, help="Number of MC samples for BNN")
    args = parser.parse_args()
    
    evaluate_dense(args.model, args.checkpoint, save_plots=args.visualize, num_mc_samples=args.mc_samples)

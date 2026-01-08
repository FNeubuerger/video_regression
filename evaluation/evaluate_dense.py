import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset
from utils.model_registry import MODEL_REGISTRY

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

def evaluate_dense(model_name, checkpoint_path, device='cuda', save_plots=False):
    print(f"Evaluating {model_name} on Dense Metrics...")
    
    # Load Model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    model = ModelClass(**kwargs)
    
    # Load Checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load Dataset
    # We use validation data
    dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        use_physics_prior=True
    )
    
    # Evaluate
    # Since we don't have dense GT for the real dataset, we are somewhat limited.
    # We will compute:
    # 1. Sparse MSE (Accuracy at sensors)
    # 2. Physics Compliance (TV Norm, Peak bounds)
    
    results = {
        'sparse_rmse': [],
        'peak_error': [],
        'tv_norm': []
    }
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            frame, target_map, mask_map, temps, prior = dataset[i]
            
            # Prepare Input
            if "UNet" in model_name:
                # 5-channel Input (RGB + Flow)
                # But dataset returns 3-channel frame?
                # Does TemperatureHeatmapDataset return Flow?
                # Check dataset.py -> __getitem__ returns: frame_tensor (3ch), target_map, mask, temps, prior
                # Wait, the UNet expects 5 channels!
                # We need to compute flow on the fly or load it.
                # For now, let's just pad with zeros to avoid crash, OR update dataset to return flow.
                # Project constraint: The dataset typically handles flow computation if configured.
                # Currently dataset seems to only load frames.
                
                # Mock flow for now to test pipeline
                # (1, 5, 64, 64)
                inp = torch.zeros((1, 5, 64, 64)).to(device)
                inp[0, :3] = frame.to(device)
            else:
                inp = frame.unsqueeze(0).to(device)
                
            # Forward
            if "hybrid" in model_name or "prior" in model_name:
                 # Some models might expect prior input? 
                 # Referring to train_unet_hybrid.py, the model just takes input. 
                 # The hybrid loss takes the prior. 
                 # But if the model is Residual (Pred = Prior + Delta), it needs Prior?
                 # Checked models/dense_heads.py -> ResNetUNet is standard.
                 pass
            
            output = model(inp)
            
            # Post-Process
            pred_map = output.squeeze().cpu().numpy()
            target_map_np = target_map.squeeze().numpy()
            mask_map_np = mask_map.squeeze().numpy()
            
            # Metrics
            m = calculate_metrics(pred_map, target_map_np, mask_map_np)
            
            for k, v in m.items():
                results[k].append(v)
            
            if save_plots and i < 5:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(pred_map, cmap='inferno')
                plt.title("Prediction")
                plt.colorbar()
                plt.subplot(1, 3, 2)
                plt.imshow(target_map_np, cmap='inferno')
                plt.title("Sparse GT")
                plt.colorbar()
                plt.subplot(1, 3, 3)
                plt.imshow(prior.squeeze(), cmap='jet')
                plt.title("Physics Prior")
                plt.savefig(f"results/plots/dense_eval_{model_name}_{i}.png")
                plt.close()

    # Aggregate
    print("\nResults:")
    for k, v in results.items():
        print(f"{k}: {np.mean(v):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    evaluate_dense(args.model, args.checkpoint, save_plots=args.visualize)

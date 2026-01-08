import torch
import numpy as np
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
import cv2
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset
from utils.model_registry import MODEL_REGISTRY
from utils.xai_wrappers import RegressionWrapper

try:
    import quantus
    from captum.attr import IntegratedGradients
except ImportError:
    print("XAI libraries not installed. Please run: pip install quantus captum")
    sys.exit(1)

def load_video_frames(sequence_path, num_samples, channels=3, target_size=(64, 64)):
    """Robust data loader that pulls frames from a sequence directory."""
    frames = []
    
    if not os.path.exists(sequence_path):
        print(f"Warning: {sequence_path} not found. Using random noise.")
        return np.random.randn(num_samples, channels, *target_size).astype(np.float32)

    files = sorted([f for f in os.listdir(sequence_path) if f.endswith('.png')])
    if not files:
        print(f"Warning: No images in {sequence_path}. Using random noise.")
        return np.random.randn(num_samples, channels, *target_size).astype(np.float32)
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Stratified sampling if we have enough frames
    indices = np.linspace(0, len(files)-1, num_samples, dtype=int)
    
    for i in indices:
        path = os.path.join(sequence_path, files[i])
        img = cv2.imread(path)
        if img is None: continue
        
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # To Tensor & Normalize
        tensor = transforms.ToTensor()(img) # (3, H, W)
        tensor = normalize(tensor)
        
        # Handle channel mismatch
        if channels == 5:
            padded = torch.zeros((5, *target_size))
            padded[:3] = tensor
            tensor = padded
        elif channels == 3:
            pass # Already 3
            
        frames.append(tensor.numpy())
        
    if len(frames) == 0:
        return np.random.randn(num_samples, channels, *target_size).astype(np.float32)
        
    return np.array(frames)

def evaluate_xai_metrics(model_name, checkpoint_path, device='cuda', num_samples=20):
    print(f"Benchmarking XAI Methods for {model_name}...")
    
    # 1. Setup Model
    if model_name not in MODEL_REGISTRY: raise ValueError(f"Model {model_name} not found")
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    
    # Smart Checkpoint Loading
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Auto-detect channels from weights
        detected_channels = 3
        if 'base_model.conv1.weight' in state_dict:
            w = state_dict['base_model.conv1.weight']
            detected_channels = w.shape[1]
        elif 'backbone.conv1.weight' in state_dict:
             w = state_dict['backbone.conv1.weight']
             detected_channels = w.shape[1]
             
        print(f"Detected model input channels: {detected_channels}")
            
        # Update config
        if 'n_channels' in kwargs: kwargs['n_channels'] = detected_channels
        if 'frame_shape' in kwargs:
             fs = list(kwargs['frame_shape'])
             if len(fs) == 3:
                 fs[2] = detected_channels
                 kwargs['frame_shape'] = tuple(fs)
        
        # Variational check
        is_variational = any("bottleneck.conv_mu" in k for k in state_dict.keys()) or \
                         any("bayesian" in k for k in state_dict.keys())
        if is_variational and 'variational' in kwargs: kwargs['variational'] = True

        model = ModelClass(**kwargs)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()
    
    # 2. Wrapper setup for XAI
    # RegressionWrapper ensures output is (B, 1) or scalar for consumption
    wrapper = RegressionWrapper(model, target_mode='mean')
    wrapper.to(device)
    wrapper.eval()
    
    # 3. Data Loading
    print(f"Loading {num_samples} samples from data/sequence_1...")
    x_batch = load_video_frames('data/sequence_1', num_samples, channels=detected_channels)
    
    # Generate model-predicted labels (Faithfulness checks against model belief, not GT)
    with torch.no_grad():
        t_batch = torch.tensor(x_batch).to(device)
        y_batch = wrapper(t_batch).cpu().numpy().flatten()
    
    print(f"Data shape: {x_batch.shape}, Labels shape: {y_batch.shape}")

    # 4. Define Explanation Function (Integrated Gradients)
    def explain_func(model, inputs, targets, **kwargs):
        # inputs: (N, C, H, W) numpy
        inputs_tensor = torch.tensor(inputs).to(device)
        inputs_tensor.requires_grad = True
        
        ig = IntegratedGradients(model)
        # target=None for scalar regression output often works or 0 if index required
        # Captum regression: usually target is not needed if output is 1D scalar per batch item
        attr = ig.attribute(inputs_tensor, target=None)
        return attr.detach().cpu().numpy()

    # 5. Metrics Definition
    # We select metrics that demonstrate:
    # 1. Faithfulness (Does heatmap reflect model logic?)
    # 2. Robustness (Is heatmap stable under noise?)
    # 3. Complexity (Is heatmap simple enough for humans?)
    
    def debug_perturb(arr, indices, indexed_axes, **kwargs):
        # Debug wrapper
        print(f"DEBUG: arr.ndim={arr.ndim}, indexed_axes={indexed_axes}")
        return quantus.perturb_func.baseline_replacement_by_indices(arr, indices, indexed_axes=indexed_axes, **kwargs)

    metrics = {
        # Faithfulness: Correlation between pixel importance and prediction drop
        "Faithfulness Correlation": quantus.FaithfulnessCorrelation(
            nr_runs=10, 
            subset_size=224, 
            perturb_baseline="black",
            perturb_func=debug_perturb,
            perturb_func_kwargs={"indexed_axes": (0, 1)},
            similarity_func=quantus.similarity_func.correlation_pearson,
            disable_warnings=True
        ),
        # Robustness: Stability of explanation under local perturbations
        "Robustness Local Lipschitz": quantus.LocalLipschitzEstimate(
            nr_samples=5,
            perturb_std=0.1,
            perturb_mean=0.0,
            similarity_func=quantus.similarity_func.distance_euclidean,
            disable_warnings=True
        ),
        # Complexity: Sparseness (Gini Index). Higher = More Sparse (Better for focus)
        "Complexity Sparseness": quantus.Sparseness(
            disable_warnings=True
        ),
         # Complexity: Complexity (Entropy). Lower = Simpler explanation
        "Complexity Entropy": quantus.Complexity(
            disable_warnings=True
        ),
        # Robustness: Max Sensitivity. Lower = More robust
        "Robustness Max Sensitivity": quantus.MaxSensitivity(
            nr_samples=5,
            lower_bound=0.1,
            disable_warnings=True
        ),
        # Axiomatic: Completeness (Sum of attributions = Output - Baseline)
        # Requires baseline=0 (black) which is implicit in IG default here
        "Axiomatic Completeness": quantus.Completeness(
            disable_warnings=True
        )
    }
    
    results = {}
    
    # 6. Run Evaluation
    # For Quantus metrics (which assume classification), we treat regression as 1-class classification.
    # We pass label=0 for all samples, and ensure model outputs (B, 1).
    y_quantus = np.zeros(num_samples, dtype=int)

    for metric_name, metric in metrics.items():
        print(f"Running {metric_name}...")
        try:
            score = metric(
                model=wrapper,
                x_batch=x_batch,
                y_batch=y_quantus, # Pass integer indices for Quantus
                channel_first=True,
                explain_func=explain_func
            )
            
            # Aggregate score
            avg_score = np.nanmean(score)
            results[metric_name] = avg_score
            print(f"-> {metric_name}: {avg_score:.4f}")
            
        except Exception as e:
            print(f"Failed {metric_name}: {e}")
            import traceback
            traceback.print_exc()
            results[metric_name] = np.nan

    # 7. Save Results
    results['Model'] = model_name
    df = pd.DataFrame([results])
    
    print("\nBenchmark Results:")
    print(df.to_string())
    
    output_dir = 'results/xai'
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, 'xai_benchmark_results.csv')
    df.to_csv(out_file, mode='a', header=not os.path.exists(out_file), index=False)
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SimpleResNet", help="Model name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simple_resnet_model.pth", help="Path to checkpoint")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    args = parser.parse_args()
    
    evaluate_xai_metrics(args.model, args.checkpoint, num_samples=args.samples)

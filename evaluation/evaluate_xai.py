import torch
import numpy as np
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset
from utils.model_registry import MODEL_REGISTRY
from utils.xai_wrappers import RegressionWrapper

try:
    import quantus
    from captum.attr import IntegratedGradients, LayerGradCam
except ImportError:
    print("XAI libraries not installed.")
    sys.exit(1)

def evaluate_xai_metrics(model_name, checkpoint_path, device='cuda', num_samples=20):
    print(f"Benchmarking XAI Methods for {model_name}...")
    
    # 1. Setup Model
    if model_name not in MODEL_REGISTRY: raise ValueError(f"Model {model_name} not found")
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    try:
        model = ModelClass(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        input_channels = kwargs.get('n_channels', 3)
    except:
        kwargs['n_channels'] = 3
        model = ModelClass(**kwargs)
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        input_channels = 3
        
    model.to(device)
    model.eval()
    
    # 2. Wrap
    wrapper = RegressionWrapper(model, target_mode='mean')
    wrapper.to(device)
    
    # 3. Data
    dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        use_physics_prior=True
    )
    
    # Select subset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    inputs_batch = []
    targets_batch = []
    
    for idx in indices:
        frame, _, _, _, _ = dataset[idx]
        if input_channels == 5:
            inp = torch.zeros((5, 64, 64))
            inp[:3] = frame
        else:
            inp = frame
        inputs_batch.append(inp.numpy())
        targets_batch.append(0) # Standard dummy target for Quantus regression/scalar wrapper
        
    x_batch = np.array(inputs_batch) # (N, C, H, W)
    y_batch = np.array(targets_batch)
    
    # Convert for Quantus (Quantus expects numpy usually, or torch depending on config)
    # We will use the model_wrapper as the model function for Quantus
    
    # Define Explanation Function (Captum wrapper)
    def explain_func(model, inputs, targets, **kwargs):
        # inputs: (N, C, H, W) numpy
        inputs_tensor = torch.tensor(inputs).to(device)
        inputs_tensor.requires_grad = True
        
        ig = IntegratedGradients(model)
        attr = ig.attribute(inputs_tensor, target=0)
        return attr.detach().cpu().numpy()
        
    # Metrics
    metrics = {
        "Faithfulness": quantus.FaithfulnessCorrelation(
            nr_runs=10, 
            subset_size=224, # Pixel subset
            perturb_baseline="black",
            return_aggregate=True
        ),
        "Robustness": quantus.LocalLipschitzEstimate(
            nr_samples=5,
            perturb_std=0.1,
            perturb_mean=0.0,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            return_aggregate=True
        )
    }
    
    results = {}
    
    print("Running Quantus Evaluation...")
    for metric_name, metric_func in metrics.items():
        print(f"Computing {metric_name}...")
        try:
            scores = metric_func(
                model=wrapper,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=None, # Quantus will call explain_func to get attributions
                explain_func=explain_func, 
                device=device
            )
            # Scores might be list or float
            if isinstance(scores, list):
                score_avg = np.mean(scores)
            else:
                score_avg = scores
                
            results[metric_name] = score_avg
            print(f"  -> {metric_name}: {score_avg:.4f}")
        except Exception as e:
            print(f"  -> Failed: {e}")
            results[metric_name] = np.nan
            
    # Save Results
    df = pd.DataFrame([results])
    df.to_csv("results/xai_benchmark_results.csv", index=False)
    print("Quantus Benchmarking Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_xai_metrics(args.model, args.checkpoint)

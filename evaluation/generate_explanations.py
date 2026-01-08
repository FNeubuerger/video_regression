import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset
from utils.model_registry import MODEL_REGISTRY
from utils.xai_wrappers import RegressionWrapper

try:
    from captum.attr import LayerGradCam, IntegratedGradients, NoiseTunnel
    from captum.attr import visualization as viz
except ImportError:
    print("Captum not installed. Please run 'pip install captum'")
    sys.exit(1)

def normalize_attr(attr):
    # Normalize to [0, 1] for visualization
    m = attr.abs().max()
    if m == 0: return attr
    return attr / m

def generate_explanations(model_name, checkpoint_path, output_dir="results/xai", device='cuda', sample_idx=0):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    
    # Load Checkpoint logic to auto-detect parameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    # Auto-detect variational
    is_variational = any("bottleneck.conv_mu" in k for k in state_dict.keys())
    if is_variational:
         print("Detected Variational Checkpoint. Enabling variational=True.")
         kwargs['variational'] = True
         
    # Auto-detect channels
    # Check conv1 weight shape in state_dict
    if 'base_model.conv1.weight' in state_dict:
        w = state_dict['base_model.conv1.weight']
        if w.shape[1] == 3:
             print("Detected 3-channel checkpoint. Overriding n_channels=3.")
             kwargs['n_channels'] = 3
        elif w.shape[1] == 5:
             kwargs['n_channels'] = 5

    try:
        model = ModelClass(**kwargs)
        model.load_state_dict(state_dict)
        input_channels = kwargs.get('n_channels', 3)
    except Exception as e:
        print(f"Instantiation failed: {e}")
        # Last ditch effort
        kwargs['n_channels'] = 3
        kwargs['variational'] = False
        model = ModelClass(**kwargs)
        model.load_state_dict(state_dict)
        input_channels = 3
        
    model.to(device)
    model.eval()
    
    # 2. Wrap Model
    # We want to explain high temperature regions.
    wrapper = RegressionWrapper(model, target_mode='mean') 
    wrapper.to(device)
    wrapper.eval()
    
    # 3. Load Data
    dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        use_physics_prior=True
    )
    
    frame, target_map, mask_map, temps, prior = dataset[sample_idx]
    
    # Prepare Input
    if input_channels == 5:
        # Pad if needed
        inp = torch.zeros((1, 5, 64, 64)).to(device)
        inp[0, :3] = frame.to(device)
        # Mock Flow (random noise or zeroes?) Zeroes for now
    else:
        inp = frame.unsqueeze(0).to(device)
        
    # Enable gradients for IG
    inp.requires_grad = True
    
    print(f"Generating explanations for {model_name} (Sample {sample_idx})...")
    
    # 4. Integrated Gradients (Input Attribution)
    ig = IntegratedGradients(wrapper)
    # Baseline: Black image
    baseline = torch.zeros_like(inp).to(device)
    
    attributions_ig, delta = ig.attribute(inp, baseline, target=0, return_convergence_delta=True)
    print(f"IG Convergence Delta: {delta}")
    
    # 5. Layer GradCAM (Feature Attribution)
    # We target the last encoder layer of the ResNet
    if hasattr(model, 'enc_layer4'):
        layer = model.enc_layer4
    elif hasattr(model, 'layer4'): # Standard ResNet
        layer = model.layer4
    else:
        print("Warning: Could not find suitable layer for GradCAM. Skipping.")
        layer = None
        
    if layer:
        gradcam = LayerGradCam(wrapper, layer)
        # GradCAM requires scalar target too
        attributions_gc = gradcam.attribute(inp, target=0)
        # Upsample to input size
        attributions_gc = LayerGradCam.interpolate(attributions_gc, (64, 64), interpolate_mode='bilinear')
    else:
        attributions_gc = None
        
    # 6. Visualization
    # Convert inputs to numpy for plotting
    # RGB
    img_rgb = inp[0, :3].permute(1, 2, 0).detach().cpu().numpy()
    # Denormalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = std * img_rgb + mean
    img_rgb = np.clip(img_rgb, 0, 1)
    
    # IG Heatmap (Sum across channels)
    attr_ig_np = np.sum(attributions_ig.squeeze().detach().cpu().numpy(), axis=0)
    attr_ig_np = np.abs(attr_ig_np) # Magnitude
    
    # Save
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Input Frame")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(target_map.squeeze(), cmap='inferno')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(attr_ig_np, cmap='jet', alpha=0.9)
    # Overlay on original?
    plt.imshow(img_rgb, alpha=0.3)
    plt.title("Integrated Gradients")
    plt.axis('off')
    
    if attributions_gc is not None:
        attr_gc_np = attributions_gc.squeeze().detach().cpu().numpy()
        # GradCAM is 1 channel usually (Conv filter weights)
        if attr_gc_np.ndim > 2: attr_gc_np = attr_gc_np.mean(axis=0)
        
        plt.subplot(1, 4, 4)
        plt.imshow(attr_gc_np, cmap='jet', alpha=0.9)
        plt.imshow(img_rgb, alpha=0.3)
        plt.title("GradCAM (Encoder L4)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"xai_{model_name}_sample{sample_idx}.png"))
    plt.close()
    print(f"Saved visualization to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sample", type=int, default=10)
    args = parser.parse_args()
    
    generate_explanations(args.model, args.checkpoint, sample_idx=args.sample)

"""
Script to visualize the learned spatial physics maps (alpha, beta) from the 
SpatialPhysicsCNNLSTM model. This provides interpretability by showing
what physical properties the model has inferred for different regions.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
from torchvision import transforms

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
from physics.models import SpatialPhysicsCNNLSTM

def visualize_maps(checkpoint_path, data_dir="data", output_dir="results/plots/physics_maps"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Hyperparameters (must match training)
    frame_shape = (64, 64, 5)
    time_steps = 5
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 3. Load Dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TemperatureSequenceDataset(
        data_dir=data_dir,
        sequence_length=time_steps,
        transform=transform,
        use_optical_flow=True,
        image_size=(64, 64),
        use_artifact_masking=True
    )
    
    # 4. Process Samples
    # Pick a few diverse samples (one early, one middle, one late sequence sequence)
    indices = [100, 1000, 5000, 10000]
    
    for idx in indices:
        if idx >= len(dataset): continue
        
        imgs, target, mask = dataset[idx]
        imgs_batch = imgs.unsqueeze(0).to(device) # (1, T, C, H, W)
        
        with torch.no_grad():
            temp_map, alpha_map, beta_map = model(imgs_batch)
            
        # temp_map (1, T, 4, 4)
        # alpha_map (1, 1, 4, 4)
        # beta_map (1, 1, 4, 4)
        
        # Convert to numpy for plotting
        t_final = temp_map[0, -1].cpu().numpy()
        alpha = alpha_map[0, 0].cpu().numpy()
        beta = beta_map[0, 0].cpu().numpy()
        
        # Original Image (last frame of sequence)
        img_unnorm = imgs[-1].cpu().permute(1, 2, 0).numpy()
        # Simple denorm for visualization
        img_unnorm = img_unnorm * 0.229 + 0.485
        img_unnorm = np.clip(img_unnorm, 0, 1)
        
        # Mask
        mask_np = mask.cpu().numpy()
        
        # Plotting
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        axes[0].imshow(img_unnorm)
        axes[0].set_title("Input (Frame T)")
        axes[0].axis('off')
        
        im1 = axes[1].imshow(t_final, cmap='hot')
        axes[1].set_title("Pred Temp $T/K$")
        plt.colorbar(im1, ax=axes[1], label='K')
        
        im2 = axes[2].imshow(alpha, cmap='viridis')
        axes[2].set_title("Perfusion $\\alpha / s^{-1}$")
        plt.colorbar(im2, ax=axes[2], label='$s^{-1}$')
        
        im3 = axes[3].imshow(beta, cmap='plasma')
        axes[3].set_title("Conductivity $\\beta / m^2 s^{-1}$")
        plt.colorbar(im3, ax=axes[3], label='$m^2 s^{-1}$')
        
        axes[4].imshow(mask_np, cmap='gray')
        axes[4].set_title("Artifact Mask")
        axes[4].axis('off')
        
        sample_name = f"sample_{idx}"
        plt.savefig(f"{output_dir}/{sample_name}.png", bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to {output_dir}/{sample_name}.png")

if __name__ == "__main__":
    cp = "models/masked/spatial_physics_cnnlstm_model.pth"
    if os.path.exists(cp):
        visualize_maps(cp)
    else:
        print(f"Checkpoint {cp} not found yet. Run after training completions.")


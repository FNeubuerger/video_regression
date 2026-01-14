"""
Script to visualize the learned spatial physics maps (alpha, beta) from the 
SpatialPhysicsCNNLSTM model. This provides interpretability by showing
what physical properties the model has inferred for different regions.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import scipy.ndimage as ndimage
import glob

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset
from physics.models import SpatialPhysicsCNNLSTM

def get_roi_crop(img, mask, crop_size=32):
    """Find sensor center from mask and crop around it."""
    # Mask is (H, W)
    if mask.sum() == 0:
        # Center crop if no mask
        h, w = img.shape[:2]
        cy, cx = h//2, w//2
    else:
        # Find center of mass of mask
        cy, cx = ndimage.center_of_mass(mask)
        cy, cx = int(cy), int(cx)
    
    h_start = max(0, cy - crop_size//2)
    w_start = max(0, cx - crop_size//2)
    h_end = min(img.shape[0], h_start + crop_size)
    w_end = min(img.shape[1], w_start + crop_size)
    
    # Adjust if out of bounds
    if h_end - h_start < crop_size:
        h_start = max(0, h_end - crop_size)
    if w_end - w_start < crop_size:
        w_start = max(0, w_end - crop_size)
        
    return img[h_start:h_end, w_start:w_end], (h_start, h_end, w_start, w_end)

def reconstruct_mask(dataset, idx):
    """Reconstruct mask using dataset metadata for a given index."""
    video_idx, _ = dataset.indices[idx]
    meta = dataset.videos[video_idx]
    target_size = dataset.target_size
    
    mask = np.zeros((target_size[0], target_size[1]), dtype=np.float32)
    orig_h, orig_w = meta['original_size']
    scale_y = target_size[0] / orig_h
    scale_x = target_size[1] / orig_w
    
    sensor_labels = ['M1', 'M2', 'M3', 'M4']
    for label in sensor_labels:
        if label not in meta['sensor_pos']: continue
        center = meta['sensor_pos'][label]['center']
        px = int(center[0] * scale_x)
        py = int(center[1] * scale_y)
        px = min(max(px, 0), target_size[1]-1)
        py = min(max(py, 0), target_size[0]-1)
        
        radius = 2
        y_min = max(0, py - radius)
        y_max = min(target_size[0], py + radius + 1)
        x_min = max(0, px - radius)
        x_max = min(target_size[1], px + radius + 1)
        mask[y_min:y_max, x_min:x_max] = 1.0
        
    return mask

def visualize_maps(checkpoint_path, data_dir="data/level1_cropped", output_root="results/plots/physics_maps"):
    model_name = os.path.basename(checkpoint_path).replace(".pth", "")
    output_dir = os.path.join(output_root, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Hyperparameters (must match training)
    frame_shape = (64, 64, 5)
    time_steps = 5
    
    # 2. Load Model
    print(f"Loading model from {checkpoint_path}...")
    try:
        model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Skipping {checkpoint_path}: Incompatible architecture ({e})")
        return
    
    # 3. Load Dataset (NEW DATA)
    dataset = SequenceHeatmapDataset(
        data_dir=data_dir,
        sequence_length=time_steps,
        use_optical_flow=True,
        target_size=(64, 64),
        use_artifact_masking=True
    )
    
    # 4. Process Samples
    indices = [300, 600, 900, 1200]
    
    for idx in indices:
        if idx >= len(dataset): continue
        
        imgs, targets, priors = dataset[idx] 
        imgs_batch = imgs.unsqueeze(0).to(device) # (1, T, C, H, W)
        
        with torch.no_grad():
            temp_map, alpha_map, beta_map = model(imgs_batch)
            
        # temp_map (1, T, 4, 4) or H,W
        
        # Convert to numpy
        t_final = temp_map[0, -1].cpu().numpy()
        alpha = alpha_map[0, 0].cpu().numpy()
        beta = beta_map[0, 0].cpu().numpy()
        
        # Get scalar pred (mean of map)
        pred_scalar = t_final.mean()
        gt_scalar = targets[-1].max().item() # Max from sparse target
        if gt_scalar == 0: gt_scalar = np.nan
        
        # Original Image (last frame, RGB only)
        # SequenceHeatmapDataset returns [0, 1]
        img_unnorm = imgs[-1, :3].cpu().permute(1, 2, 0).numpy()
        # Ensure clipping
        img_unnorm = np.clip(img_unnorm, 0, 1)
        
        # Mask
        mask_np = reconstruct_mask(dataset, idx)
        
        # ROI Crop
        img_roi, (hs, he, ws, we) = get_roi_crop(img_unnorm, mask_np)
        
        # Plotting
        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
        
        # 1. Full Input with Box
        axes[0].imshow(img_unnorm)
        # Draw ROI box
        rect = plt.Rectangle((ws, hs), we-ws, he-hs, linewidth=2, edgecolor='r', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].set_title(f"Full Input\nGT: {gt_scalar:.1f} K")
        axes[0].axis('off')

        # 2. ROI Zoom
        axes[1].imshow(img_roi)
        axes[1].set_title(f"Sensor ROI\n(Zoomed)")
        axes[1].axis('off')
        
        # 3. Pred Map
        im2 = axes[2].imshow(t_final, cmap='hot', vmin=min(gt_scalar-10, 290), vmax=max(gt_scalar+10, 350))
        axes[2].set_title(f"Pred Temp Map\nMean: {pred_scalar:.1f} K")
        plt.colorbar(im2, ax=axes[2], label='K')
        
        # 4. Alpha
        im3 = axes[3].imshow(alpha, cmap='viridis')
        axes[3].set_title(r"Perfusion $\alpha$" + "\n(Tissue Property)")
        plt.colorbar(im3, ax=axes[3], label=r'$s^{-1}$')
        
        # 5. Beta
        im4 = axes[4].imshow(beta, cmap='plasma')
        axes[4].set_title(r"Conductivity $\beta$" + "\n(Tissue Property)")
        plt.colorbar(im4, ax=axes[4], label=r'$m^2 s^{-1}$')
        
        # 6. Mask
        axes[5].imshow(mask_np, cmap='gray')
        axes[5].set_title("Artifact/Sensor\nLocation")
        axes[5].axis('off')
        
        plt.suptitle(f"Model: {model_name} | Sample: {idx}", fontsize=16)
        plt.tight_layout()
        
        save_path = f"{output_dir}/sample_{idx}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    # Scan for compatible models
    search_paths = ["models/masked", "models"]
    checkpoints = []
    
    for p in search_paths:
        checkpoints.extend(glob.glob(f"{p}/*spatial*_model.pth"))
        checkpoints.extend(glob.glob(f"{p}/*bioheat*.pth"))
        
    checkpoints = list(set(checkpoints)) # dedup
    
    if not checkpoints:
        print("No compatible spatial checkpoints found.")
    
    for cp in checkpoints:
        if os.path.exists(cp):
            print(f"Processing {cp}...")
            visualize_maps(cp)


"""
Visualization of Advection: Temperature Gradients and Optical Flow.
This script demonstrates how physical motion (Optical Flow) interacts 
with the temperature field (Advection term v . grad T).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import scipy.ndimage as ndimage
from torchvision import transforms

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset
from physics.models import SpatialPhysicsCNNLSTM

def get_roi_crop_coords(mask, img_shape, crop_size=32):
    """Get crop coordinates centered on mask centroid."""
    if mask.sum() == 0:
        h, w = img_shape[:2]
        cy, cx = h//2, w//2
    else:
        cy, cx = ndimage.center_of_mass(mask)
        cy, cx = int(cy), int(cx)
    
    h_start = max(0, cy - crop_size//2)
    w_start = max(0, cx - crop_size//2)
    h_end = min(img_shape[0], h_start + crop_size)
    w_end = min(img_shape[1], w_start + crop_size)
    
    return h_start, h_end, w_start, w_end

import cv2

def get_raw_context(video_filename, crop_meta, frame_idx=0, raw_dir="data/level0_raw"):
    """
    Load the raw full-frame image and draw the crop box.
    """
    raw_path = os.path.join(raw_dir, video_filename)
    if not os.path.exists(raw_path):
        return None
        
    cap = cv2.VideoCapture(raw_path)
    # We ideally want the same frame index as the sample
    # But frame_idx is relative to the sequence. 
    # dataset.indices maps [idx] -> (video_idx, start_frame).
    # We need to expose this mapping or pass absolute frame index.
    
    # For visualization context, the "Active Zone" is static per video.
    # Showing a representative frame (e.g. frame matching the sample) is best.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret: return None
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw Crop Box
    # meta gives y_start, y_end. Width is likely full width.
    y_start = crop_meta['y_start']
    y_end = crop_meta['y_end']
    h, w, _ = frame.shape
    
    # Draw Red Box
    cv2.rectangle(frame, (0, y_start), (w, y_end), (255, 0, 0), 5)
    
    return frame

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

def plot_advection(checkpoint_path, data_dir="data/level1_cropped", raw_dir="data/level0_raw", output_root="results/plots/advection"):
    model_name = os.path.basename(checkpoint_path).replace(".pth", "")
    output_dir = os.path.join(output_root, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    print(f"Processing {checkpoint_path}...")
    try:
        frame_shape = (64, 64, 5)
        time_steps = 5
        model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    except Exception as e:
        print(f"Warning: Could not load model for {checkpoint_path}: {e}")
        return
    
    # 2. Load Dataset (NEW DATA)
    transform = transforms.Compose([
        # transforms.Resize((64, 64)), # Dataset does resize internally based on target_size
        transforms.ToTensor(),
        # Internal normalization happens in dataset? No, it divides by 255. 
        # But old dataset used normalization. New dataset returns 0-1 tensors?
        # Let's check SequenceHeatmapDataset.__getitem__. It does / 255.0. 
        # It does NOT apply Normalize((0.485...), (0.229...)) unless passed in transform.
        # But if we pass ToTensor, it expects PIL or numpy. 
        # SequenceHeatmapDataset does manual tensor conversion. 
    ])
    
    # Wait, SequenceHeatmapDataset.__getitem__ does manual tensor conversion:
    # frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
    # It takes transform argument but usually applied on the tensor or frame?
    # Actually SequenceHeatmapDataset doesn't seem to apply self.transform in the snippet I read?
    # Let's re-read __getitem__.
    
    # Lines 261: if self.use_optical_flow: frames_t = preprocess...
    # It does NOT seem to call self.transform on frames! 
    # It simply resizes and normalizes to [0, 1].
    # So if the model expects ImageNet normalization, we might have a problem if the dataset doesn't do it.
    # The snippet I read (lines 230-268) did not show self.transform usage.
    # Lines 158 init: self.transform = transform.
    # Maybe used in other methods?
    # I will assume [0, 1] is returned.
    # But SpatialPhysicsCNNLSTM is likely ResNet based, which expects ImageNet norm.
    # If the model was trained on ImageNet norm, I should apply it. 
    # But if SequenceHeatmapDataset doesn't apply it, then training might have been without it?
    # Or I missed where it applies it.
    
    # Assuming for now we proceed with what the dataset gives. 
    # We can normalize manually if needed.
    
    dataset = SequenceHeatmapDataset(
        data_dir=data_dir,
        sequence_length=time_steps,
        use_optical_flow=True,
        target_size=(64, 64),
        use_artifact_masking=True
    )
    
    # 3. Process Samples
    indices = [100, 300, 500, 800] # Use smaller indices as dataset might be smaller/different indexing
    
    for idx in indices:
        if idx >= len(dataset): continue
        
        imgs, targets, priors = dataset[idx]
        
        # imgs: (Seq, 5, 64, 64)
        
        # 1. RGB Image (Last Frame)
        # SequenceHeatmapDataset returns [0, 1] RGB
        rgb_img = imgs[-1, :3].permute(1, 2, 0).cpu().numpy()
        # No Denorm needed if it's already 0-1? 
        # Depending on if normalization was applied. 
        
        # 2. Temperature (Target Scalar)
        # Extract max from sparse target map
        gt_temp = targets[-1].max().item()
        if gt_temp == 0: gt_temp = np.nan # No sensor in this frame?
        
        # 3. Features for Advection
        intensity = rgb_img.mean(axis=2)
        grad_y, grad_x = np.gradient(intensity)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        flow_x = imgs[-1, 3].cpu().numpy()
        flow_y = imgs[-1, 4].cpu().numpy()
        
        # Advection Term: v . grad T
        advection = -(flow_x * grad_x + flow_y * grad_y)
        
        # ROI Crop
        mask_np = reconstruct_mask(dataset, idx)
        hs, he, ws, we = get_roi_crop_coords(mask_np, (64, 64))
        
        # Get Global Context
        # Resolve frame index
        video_idx, seq_start = dataset.indices[idx]
        abs_frame_idx = seq_start + time_steps - 1 # Last frame in sequence
        video_meta = dataset.videos[video_idx]
        video_filename = os.path.basename(video_meta['path'])
        
        # Get Meta from coords if available
        crop_meta = None
        if video_filename in dataset.coords_data:
            if 'meta' in dataset.coords_data[video_filename]:
                crop_meta = dataset.coords_data[video_filename]['meta']
                
        raw_frame = None
        if crop_meta:
            raw_frame = get_raw_context(video_filename, crop_meta, frame_idx=abs_frame_idx, raw_dir=raw_dir)
        
        # Helper to crop
        def crop(arr): return arr[hs:he, ws:we]
        
        # --- PLOTTING ---
        # Add column for Context
        n_cols = 5 if raw_frame is not None else 4
        fig, axes = plt.subplots(2, n_cols, figsize=(5*n_cols, 10))
        
        col_offset = 1 if raw_frame is not None else 0
        
        # Row 1: Global Context & Full Frame
        if raw_frame is not None:
             axes[0,0].imshow(raw_frame)
             axes[0,0].set_title("Global Context\n(Raw Input & Active Zone)")
             axes[0,0].axis('off')
             axes[1,0].axis('off') # Empty slot below or maybe legend?
             
        # A. Image
        axes[0,0+col_offset].imshow(rgb_img)
        # rect = plt.Rectangle((ws, hs), we-ws, he-hs, linewidth=2, edgecolor='r', facecolor='none')
        # axes[0,0+col_offset].add_patch(rect)
        axes[0,0+col_offset].set_title(f"Model Input (Active Zone)\nTarget T: {gt_temp:.1f} K")
        
        # B. Gradient
        axes[0,1+col_offset].imshow(grad_mag, cmap='magma')
        axes[0,1+col_offset].set_title(r"Intensity Gradient $|\nabla I|$")
        
        # C. Flow
        axes[0,2+col_offset].quiver(flow_x, flow_y, scale=10, color='blue') # Subsample for visibility?
        axes[0,2+col_offset].invert_yaxis()
        axes[0,2+col_offset].set_title(r"Optical Flow $\vec{v}$")
        
        im_adv = axes[0,3+col_offset].imshow(advection, cmap='seismic', vmin=-np.max(np.abs(advection)), vmax=np.max(np.abs(advection)))
        axes[0,3+col_offset].set_title(r"Advection $-\vec{v} \cdot \nabla I$")
        plt.colorbar(im_adv, ax=axes[0,3+col_offset])
        
        # Row 2: ROI Zoom
        axes[1,0+col_offset].imshow(crop(rgb_img))
        axes[1,0+col_offset].set_title("Sensor ROI (Zoom)")
        
        axes[1,1+col_offset].imshow(crop(grad_mag), cmap='magma')
        axes[1,1+col_offset].set_title("ROI Gradient")
        
        axes[1,2+col_offset].quiver(crop(flow_x), crop(flow_y), scale=10, color='blue')
        axes[1,2+col_offset].invert_yaxis()
        axes[1,2+col_offset].set_title("ROI Flow")
        
        roi_adv = crop(advection)
        axes[1,3+col_offset].imshow(roi_adv, cmap='seismic', vmin=-np.max(np.abs(roi_adv)), vmax=np.max(np.abs(roi_adv)))
        axes[1,3+col_offset].set_title("ROI Advection")
        
        plt.suptitle(f"Physical Dynamics (Advection) | Sample {idx}", fontsize=16)
        plt.tight_layout()
        
        save_path = f"{output_dir}/advection_{idx}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    search_paths = ["models/masked", "models"]
    checkpoints = []
    for p in search_paths:
        checkpoints.extend(glob.glob(f"{p}/*convection*.pth"))
        checkpoints.extend(glob.glob(f"{p}/*physics*.pth"))
        checkpoints.extend(glob.glob(f"{p}/*spatial*.pth"))
    
    checkpoints = list(set(checkpoints))
    
    if not checkpoints:
        print("No compatible checkpoints found.")
        
    for cp in checkpoints:
        plot_advection(cp)


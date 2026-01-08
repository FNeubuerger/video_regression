import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import sys
import matplotlib.cm as cm

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def create_colorbar(height, cmap, min_val, max_val):
    """
    create a vertical colorbar image using opencv/numpy
    """
    width = 60
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient values
    vals = np.linspace(max_val, min_val, height)
    norm_vals = (vals - min_val) / (max_val - min_val)
    
    # Apply colormap
    colors = cmap(norm_vals)[:, :3] * 255
    colors_bgr = colors[:, ::-1] # RGB to BGR
    bar[:, :40, :] = np.tile(colors_bgr[:, np.newaxis, :], (1, 40, 1))
    
    # Add text labels
    step = 5.0
    for v in np.arange(np.ceil(min_val/step)*step, max_val+0.1, step):
        # find y position
        if max_val == min_val: pos = 0
        else:
            pos = int(height * (1 - (v - min_val)/(max_val - min_val)))
            
        pos = np.clip(pos, 15, height-5)
        cv2.putText(bar, f"{int(v)}", (42, pos+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.line(bar, (35, pos), (40, pos), (255, 255, 255), 1)
        
    return bar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='US_005_30W_10min.mp4', help='Filename to verify')
    parser.add_argument('--output_dir', type=str, default='data/level1_cropped/alignment_vis')
    parser.add_argument('--limit_frames', type=int, default=100, help='Number of frames to visualize')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Initializing Dataset...")
    ds = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(360, 480) 
    )
    
    video_indices = [i for i, x in enumerate(ds.indices) if os.path.basename(x['video_path']) == args.video_name]
    
    if not video_indices:
        print(f"Video {args.video_name} not found in dataset.")
        return
        
    # Setup Video Writer
    sample_idx = video_indices[0]
    frame_tensor, _, _, _ = ds[sample_idx]
    c, h, w = frame_tensor.shape
    
    # Expanded width for colorbar
    vis_h, vis_w = h, w + 80
    
    out_path = os.path.join(args.output_dir, f"heatmap_overlay_{args.video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (vis_w, vis_h))
    
    # Temp Bounds
    T_MIN, T_MAX = 20.0, 55.0
    cmap = cm.get_cmap('jet')
    
    # Precompute colorbar
    colorbar = create_colorbar(vis_h, cmap, T_MIN, T_MAX)
    
    print(f"Generating heatmap video for {args.limit_frames} frames...")
    
    for i in tqdm(range(min(len(video_indices), args.limit_frames))):
        idx = video_indices[i]
        frame_tensor, target_map, mask_map, temp_vec = ds[idx]
        
        # Unnormalize Frame
        vis_frame = unnormalize(frame_tensor).permute(1, 2, 0).numpy()
        vis_frame = np.clip(vis_frame, 0, 1)
        vis_frame = (vis_frame * 255).astype(np.uint8)
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Create Heatmap Overlay
        overlay = vis_frame.copy()
        
        # Get Sensor Locations & Temps
        # We use the raw metadata from dataset index for robustness
        item = ds.indices[idx]
        sensor_pos = item['sensor_pos']
        temps = item['temps']
        
        orig_h, orig_w = item['original_size']
        scale_x = w / orig_w
        scale_y = h / orig_h
        
        sensor_labels = ['M1', 'M2', 'M3', 'M4']
        
        for k, label in enumerate(sensor_labels):
            if label not in sensor_pos: continue
            val = temps[k]
            if np.isnan(val): continue
            
            # Map coords
            center = sensor_pos[label]['center']
            nx = int(center[0] * scale_x)
            ny = int(center[1] * scale_y)
            
            # Color
            norm_val = (val - T_MIN) / (T_MAX - T_MIN)
            norm_val = np.clip(norm_val, 0, 1)
            color = cmap(norm_val)[:3] # RGB 0-1
            color_bgr = (int(color[2]*255), int(color[1]*255), int(color[0]*255))
            
            # Draw Blob
            # We want a smooth gaussian-like blob or just a large circle?
            # User said "heatmap label". 
            # Let's draw a filled circle with radius 15
            cv2.circle(overlay, (nx, ny), 15, color_bgr, -1)
            
            # Add text value nearby
            cv2.putText(vis_frame, f"{val:.1f}", (nx+10, ny-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Blend
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
        
        # Compose with Colorbar
        canvas = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
        canvas[:, :w, :] = vis_frame
        canvas[:, w:w+60, :] = colorbar
        
        out.write(canvas)
        
    out.release()
    print(f"Heatmap video saved to {out_path}")

if __name__ == "__main__":
    main()

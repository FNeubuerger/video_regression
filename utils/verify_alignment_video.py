import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import sys

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset

def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str, default='US_005_30W_10min.mp4', help='Filename to verify')
    parser.add_argument('--output_dir', type=str, default='data/level1_cropped/alignment_vis')
    parser.add_argument('--limit_frames', type=int, default=500, help='Number of frames to visualize')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print(f"Initializing Dataset...")
    ds = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(360, 480) # Keep a reasonable resolution for visualization
    )
    
    # Filter indices for specific video
    # The dataset flattens everything, so we need to find the range for our video
    video_indices = [i for i, x in enumerate(ds.indices) if os.path.basename(x['video_path']) == args.video_name]
    
    if not video_indices:
        print(f"Video {args.video_name} not found in dataset.")
        return
        
    print(f"Found {len(video_indices)} frames for {args.video_name}. Generating video for first {args.limit_frames}...")
    
    # Setup Video Writer
    sample_idx = video_indices[0]
    frame_tensor, _, _, _ = ds[sample_idx]
    c, h, w = frame_tensor.shape
    
    out_path = os.path.join(args.output_dir, f"align_{args.video_name}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
    
    for i in tqdm(range(min(len(video_indices), args.limit_frames))):
        idx = video_indices[i]
        
        # Get data from dataset
        # This calls __getitem__, which does loading, resizing, normalizing
        # This ensures we see EXACTLY what the network sees (modulo unnormalization)
        frame_tensor, target_map, mask_map, temp_vec = ds[idx]
        
        # Unnormalize
        vis_frame = unnormalize(frame_tensor).permute(1, 2, 0).numpy()
        vis_frame = np.clip(vis_frame, 0, 1)
        vis_frame = (vis_frame * 255).astype(np.uint8)
        # RGB to BGR for OpenCV
        vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
        
        # Draw Sensors and Temperatures
        # Mask map has 1s at sensor locations.
        # target_map has values.
        
        # We can use the mask to find coordinates
        ys, xs = mask_map[0].nonzero(as_tuple=True)
        
        # Since we might have multiple pixels per sensor depending on implementation (currently single pixel)
        # We iterate through the temp_vec explicitly to map to M1-M4 logic if we wanted labels
        # But visualizing the sparse map is also good.
        
        # Let's verify against the known sensor positions in the indices metadata for robustness
        item = ds.indices[idx]
        sensor_pos = item['sensor_pos']
        temps = item['temps'] 
        
        # We need to map original sensor coords to resized coords
        orig_h, orig_w = item['original_size']
        scale_x = w / orig_w
        scale_y = h / orig_h
        
        sensor_labels = ['M1', 'M2', 'M3', 'M4']
        
        for k, label in enumerate(sensor_labels):
            if label not in sensor_pos: continue
            
            temp_val = temps[k]
            if np.isnan(temp_val): continue
            
            center = sensor_pos[label]['center']
            nx = int(center[0] * scale_x)
            ny = int(center[1] * scale_y)
            
            # Draw Circle
            cv2.circle(vis_frame, (nx, ny), 5, (0, 255, 0), 2)
            
            # Draw Temperature Background
            text = f"{temp_val:.1f}C"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis_frame, (nx + 8, ny - 15), (nx + 8 + tw, ny + 5), (0, 0, 0), -1)
            
            # Draw Temperature Text
            cv2.putText(vis_frame, text, (nx + 8, ny), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                       
        out.write(vis_frame)
        
    out.release()
    print(f"Alignment verification video saved to {out_path}")

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset

def visualize_roi_context(data_dir="data/level1_cropped", raw_dir="data/level0_raw", output_dir="results/roi_check"):
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = SequenceHeatmapDataset(
        data_dir=data_dir,
        raw_dir=raw_dir,
        sequence_length=1,
        target_size=(64, 64),
        use_physics_prior=False
    )
    
    print(f"Dataset loaded. Found {len(dataset.videos)} videos.")
    
    # Process each video once
    seen_videos = set()
    
    # Iterate through dataset indices to find one sample per video
    for idx in range(0, len(dataset), 50): # Skip frames to be fast
        video_idx, start_frame = dataset.indices[idx]
        video_meta = dataset.videos[video_idx]
        video_filename = os.path.basename(video_meta['path'])
        
        if video_filename in seen_videos:
            continue
        seen_videos.add(video_filename)
        
        print(f"Visualizing ROI for {video_filename}...")
        
        # Get Meta
        crop_meta = None
        if video_filename in dataset.coords_data:
            if 'meta' in dataset.coords_data[video_filename]:
                crop_meta = dataset.coords_data[video_filename]['meta']
        
        if crop_meta is None:
            print(f"  Warning: No crop metadata for {video_filename}")
            continue
            
        # Load Raw Frame
        raw_path = os.path.join(raw_dir, video_filename)
        if not os.path.exists(raw_path):
            print(f"  Warning: Raw video not found at {raw_path}")
            continue
            
        cap = cv2.VideoCapture(raw_path)
        # Use a middle frame or start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, raw_frame = cap.read()
        cap.release()
        
        if not ret:
            print("  Failed to read raw frame.")
            continue
            
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        h_raw, w_raw, _ = raw_frame.shape
        
        # Fetch Dataset Item (Cropped)
        # dataset[idx] returns (images, labels, mask)
        # images is (C, H, W) or (T, C, H, W)
        batch = dataset[idx]
        if len(batch) == 3:
            images, labels, mask = batch
        else:
            images, labels, mask, _ = batch
            
        if images.dim() == 4: # (T, C, H, W)
            img_crop = images[0] # First frame
        else:
            img_crop = images
            
        # Convert to HWC for plotting
        # Assuming normalized, so might need denorm?
        # But for just structure check, raw tensor is ok or clip
        img_crop = img_crop[:3, :, :].permute(1, 2, 0).numpy()
        img_crop = (img_crop - img_crop.min()) / (img_crop.max() - img_crop.min())
        
        # Draw Box on Raw
        y_start = crop_meta['y_start']
        y_end = crop_meta['y_end']
        # x is full width per methodology
        x_start = 0
        x_end = w_raw
        
        vis_raw = raw_frame.copy()
        cv2.rectangle(vis_raw, (x_start, y_start), (x_end, y_end), (255, 0, 0), 10)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(vis_raw)
        axes[0].set_title(f"Raw Input (Full Frame)\nCrop: y[{y_start}:{y_end}]")
        axes[0].axis('on')
        
        axes[1].imshow(img_crop)
        axes[1].set_title(f"Network Input (Resized)\n{video_filename}")
        axes[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"roi_{video_filename[:-4]}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved to {save_path}")

if __name__ == "__main__":
    visualize_roi_context()

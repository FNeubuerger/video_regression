import cv2
import numpy as np
import os
import glob
import json
import argparse
from tqdm import tqdm
from preprocess_dataset import detect_sensors_in_frame, get_crop_bounds

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', default='data/level0_raw')
    parser.add_argument('--cropped_json', default='data/level1_cropped/sensor_coordinates.json')
    parser.add_argument('--output_dir', default='data/comparison_vis')
    args = parser.parse_args()
    
    ensure_dir(args.output_dir)
    
    # Load Level 1 Data (Ground Truth from Cropped Pipeline)
    with open(args.cropped_json, 'r') as f:
        level1_data = json.load(f)
        
    video_files = sorted(glob.glob(os.path.join(args.raw_dir, "*.mp4")))
    
    print(f"Generating ROI Comparison visualizations...")
    
    for v_path in tqdm(video_files):
        filename = os.path.basename(v_path)
        if 'verified_' in filename: continue
        
        # 1. Load Raw Frame (Average)
        cap = cv2.VideoCapture(v_path)
        frames = []
        for _ in range(15):
            ret, f = cap.read()
            if ret: frames.append(f)
        cap.release()
        
        if not frames: continue
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        
        # 2. Get Metadata for ROI
        if filename not in level1_data:
            print(f"Skipping {filename} (no level 1 data)")
            continue
            
        l1_entry = level1_data[filename]
        meta = l1_entry.get('meta', {})
        y_start = meta.get('y_start', 0)
        y_end = meta.get('y_end', avg_frame.shape[0])
        
        # 3. Detect on Raw (Naive Comparison)
        # We run the SAME detector, but on the FULL frame, to see if it gets distracted.
        raw_detections = detect_sensors_in_frame(avg_frame)
        
        # 4. Visualization
        vis = avg_frame.copy()
        h, w = vis.shape[:2]
        
        # Draw ROI (Active Zone) - Yellow Dashed
        # cv2 doesn't do dashed easily, plain rect
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, y_start), (w, y_end), (0, 255, 255), -1) 
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        cv2.rectangle(vis, (0, y_start), (w, y_end), (0, 255, 255), 2)
        cv2.putText(vis, "Active Zone (Crop ROI)", (10, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw Raw Detections (Red) - What would we get without cropping?
        if raw_detections:
            for lbl, d in raw_detections.items():
                cx, cy = d['center']
                cv2.circle(vis, (cx, cy), d['radius'] + 4, (0, 0, 255), 2)
                cv2.putText(vis, f"Raw:{lbl}", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw Final Detections (Green) - From Level 1 (shifted back)
        for lbl, d in l1_entry.items():
            if lbl == 'meta': continue
            cx_crop, cy_crop = d['center']
            
            # Map back to raw
            cx_raw = cx_crop
            cy_raw = cy_crop + y_start
            
            cv2.circle(vis, (cx_raw, cy_raw), d['radius'], (0, 255, 0), 2)
            cv2.circle(vis, (cx_raw, cy_raw), 2, (0, 255, 0), -1)
            cv2.putText(vis, f"{lbl}", (cx_raw-10, cy_raw-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Legend
        cv2.putText(vis, "Green: Cropped & Robust (Selected)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, "Red: Full Frame Detection (Comparison)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out_path = os.path.join(args.output_dir, f"comp_{filename}.jpg")
        cv2.imwrite(out_path, vis)
        
    print(f"Saved comparison images to {args.output_dir}")

if __name__ == "__main__":
    main()

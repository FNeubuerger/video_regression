import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def analyze_roi(data_dir):
    video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
    
    print(f"{'Filename':<30} {'Height':<10} {'Peak Y':<10} {'Peak Value':<10}")
    print("-" * 60)
    
    stats = []
    
    for video_path in sorted(video_files):
        filename = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            continue
            
        # Read a few frames
        frames = []
        for _ in range(10):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(gray)
            else:
                break
        cap.release()
        
        if not frames:
            continue
            
        mean_frame = np.mean(frames, axis=0)
        
        # Horizontal projection (average intensity of each row)
        # axis=1 means reduce columns, so we get size (H,)
        row_profile = np.mean(mean_frame, axis=1)
        
        # Find the row with maximum intensity (the "thick white line")
        peak_y = np.argmax(row_profile)
        peak_val = row_profile[peak_y]
        height = mean_frame.shape[0]
        
        stats.append({
            'filename': filename,
            'height': height,
            'peak_y': peak_y,
            'peak_val': peak_val
        })
        
        print(f"{filename:<30} {height:<10} {peak_y:<10} {peak_val:<10.2f}")
        
    return stats

if __name__ == "__main__":
    analyze_roi('data/new_data/BiT_Projekt')

import os
import json
import glob
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TemperatureHeatmapDataset(Dataset):
    def __init__(self, 
                 data_dir="data/level1_cropped", 
                 raw_dir="data/level0_raw", 
                 transform=None,
                 target_size=(256, 256)):
        """
        Args:
            data_dir: Directory containing cropped MP4 videos and sensor_coordinates.json
            raw_dir: Directory containing CSV log files
            transform: Optional transform to be applied on a sample.
            target_size: Tuple (H, W) to resize frames and coordinates to.
        """
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.transform = transform
        self.target_size = target_size
        
        # Load Sensor Coordinates
        json_path = os.path.join(data_dir, "sensor_coordinates.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Sensor coordinates not found at {json_path}")
            
        with open(json_path, 'r') as f:
            self.coords_data = json.load(f)
            
        self.indices = [] # List of (video_path, frame_idx, csv_row_idx, sensor_dict)
        self.video_caps = {} # Cache video info (not caps themselves)
        
        self._index_dataset()
        
    def _get_csv_path(self, video_filename):
        # US_001_30W_10min.mp4 -> LogJob_001_30W_10min.csv
        parts = video_filename.split('_')
        identifier = parts[1] # 001
        
        # 1. Try 'cleaned' subdirectory first
        cleaned_pattern = os.path.join(self.raw_dir, "cleaned", f"LogJob_{identifier}_*.csv")
        candidates = glob.glob(cleaned_pattern)
        if candidates:
            return candidates[0]
            
        # 2. Try root
        search_pattern = os.path.join(self.raw_dir, f"*{identifier}*.csv")
        candidates = glob.glob(search_pattern)
        if candidates:
            # Prefer 'LogJob'
            log_jobs = [c for c in candidates if 'LogJob' in os.path.basename(c)]
            if log_jobs:
                return log_jobs[0]
            return candidates[0]
        return None

    def _index_dataset(self):
        video_files = sorted(glob.glob(os.path.join(self.data_dir, "*.mp4")))
        print(f"Indexing {len(video_files)} videos...")
        
        for v_path in video_files:
            fname = os.path.basename(v_path)
            if 'verified_' in fname: continue
            
            # Get Coords
            if fname not in self.coords_data:
                continue
            sensor_pos = self.coords_data[fname]
            
            # Get CSV
            csv_path = self._get_csv_path(fname)
            if not csv_path or not os.path.exists(csv_path):
                print(f"Skipping {fname}: CSV not found")
                continue
                
            # Load CSV Data
            try:
                # Assuming cleaned data or robust loader. 
                # For now using pandas with separators that might be ; or ,
                df = pd.read_csv(csv_path, sep=None, engine='python')
                
                # Check columns
                req_cols = ['C26M1_Ch1', 'C26M2_Ch1', 'C26M3_Ch1', 'C26M4_Ch1']
                # If not present, might be raw file with different header
                # Try locating them
                available_cols = [c for c in req_cols if c in df.columns]
                if len(available_cols) < 4:
                    print(f"Skipping {fname}: Missing sensor columns in CSV")
                    continue
                
                temps = df[req_cols].values # (N_logs, 4)
                
            except Exception as e:
                print(f"Skipping {fname}: CSV error {e}")
                continue
                
            # Get Video Info
            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened(): continue
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            n_logs = len(temps)
            
            # Map frames
            # Logic: Video usually matches Log duration.
            # We assume linear mapping.
            for i in range(n_frames):
                # Map frame index 'i' to log index 'j'
                j = min(int(i * (n_logs / n_frames)), n_logs - 1)
                
                self.indices.append({
                    'video_path': v_path,
                    'frame_idx': i,
                    'temps': temps[j], # [T1, T2, T3, T4]
                    'sensor_pos': sensor_pos, # Dict of M1..M4
                    'original_size': (height, width)
                })
                
        print(f"Indexed {len(self.indices)} frames.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.indices[idx]
        
        # Load Frame
        cap = cv2.VideoCapture(item['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, item['frame_idx'])
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback: return zero tensor
            return torch.zeros((3, *self.target_size)), torch.zeros((4,)), torch.zeros((1, *self.target_size))

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        h, w = item['original_size']
        th, tw = self.target_size
        
        frame_resized = cv2.resize(frame, (tw, th))
        
        # Transform Frame to Tensor
        # (Usually Normalize here)
        tensor_frame = transforms.ToTensor()(frame_resized)
        tensor_frame = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_frame)
        
        # Create Sparse Target Map & Mask
        # We need to map sensor coordinates to new size
        scale_x = tw / w
        scale_y = th / h
        
        # We can create a mask where:
        # channel 0: M1 temp (at M1 pos), M2 temp (at M2 pos)...?
        # Or a single channel map with sparse values.
        
        target_map = torch.zeros((1, th, tw), dtype=torch.float32)
        mask_map = torch.zeros((1, th, tw), dtype=torch.float32)
        
        sensor_labels = ['M1', 'M2', 'M3', 'M4']
        temp_values = item['temps']
        
        for k, label in enumerate(sensor_labels):
            if label not in item['sensor_pos']: continue
            
            data = item['sensor_pos'][label]
            cx, cy = data['center']
            
            # Scale
            nx = int(cx * scale_x)
            ny = int(cy * scale_y)
            
            # Clip
            nx = min(max(nx, 0), tw - 1)
            ny = min(max(ny, 0), th - 1)
            
            # Assign
            val = float(temp_values[k])
            
            if np.isnan(val):
                continue
            
            # We can use a small radius or single pixel
            # For UNet training, maybe single pixel is hard. 
            # Let's do 3x3 block or gaussian? 
            # Issue #10 says "Sparse MSE Loss: Calculates error only at the 4 specific pixel coordinates".
            # So single pixel is fine if loss handles it.
            
            target_map[0, ny, nx] = val
            mask_map[0, ny, nx] = 1.0
            
        return tensor_frame, target_map, mask_map, torch.tensor(temp_values, dtype=torch.float32)

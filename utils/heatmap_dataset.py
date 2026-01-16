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
                 target_size=(256, 256),
                 use_physics_prior=True,
                 use_artifact_masking=False):
        """
        Args:
            data_dir: Directory containing cropped MP4 videos and sensor_coordinates.json
            raw_dir: Directory containing CSV log files
            transform: Optional transform to be applied on a sample.
            target_size: Tuple (H, W) to resize frames and coordinates to.
            use_physics_prior: If True, generates Gaussian heatmap based on wattage. If False, returns zeros.
            use_artifact_masking: If True, returns a mask identifying sensor regions to be ignored.
        """
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.transform = transform
        self.target_size = target_size
        self.use_physics_prior = use_physics_prior
        self.use_artifact_masking = use_artifact_masking
        
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

    @staticmethod
    def _parse_power_from_filename(filename):
        """
        Extracts power wattage from filename.
        US_005_30W_10min.mp4 -> 30.0
        """
        try:
            parts = filename.split('_')
            for p in parts:
                if p.endswith('W'):
                    return float(p[:-1])
        except:
            pass
        return 30.0 # Default fallback

    @staticmethod
    def _generate_physics_prior(H, W, power_watts):
        """
        Generates a 2D Gaussian heatmap centered in the frame.
        Peak intensity scales with Power.
        """
        # Physics approximation: Center heating
        # Peak temperature roughly linear with power? Let's assume proportional.
        # Base body temp is ~37 (normalized to ~0.4?).
        # Max temp at 30W is ~55C.
        # Let's create a specialized prior.
        
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Gaussian parameters
        # Center is (0,0) in grid coords
        sigma_x = 0.5 
        sigma_y = 0.5 # More spread vertically?
        
        # Gaussian decay
        dist = (xx**2)/(2*sigma_x**2) + (yy**2)/(2*sigma_y**2)
        heatmap = torch.exp(-dist)
        
        # Scale by power
        # 50W -> stronger peak. 30W -> weaker.
        # Normalize: 30W maps to value 1.0 (arbitrary)
        scale = power_watts / 30.0
        
        # Baseline body temp offset (approx 37C)
        # But we work in raw values or normalized?
        # The dataset returns raw temps in the sparse vector.
        # But we want the prior to be in degrees Celsius if we do T_pred = Prior + Delta
        
        # Let's approximate: Ambient=37C, Peak Rise = Power * Factor
        # T = 37 + (Power * 0.8) * Gaussian
        
        T_ambient = 37.0
        T_rise = power_watts * 0.6 # e.g. 30W * 0.6 = 18C rise -> 55C peak. 50W -> 30C rise -> 67C peak.
        
        prior = T_ambient + T_rise * heatmap
        return prior.unsqueeze(0) # (1, H, W)

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
            
            # Get Power
            power = self._parse_power_from_filename(fname)
            
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
            # Precompute prior for this video
            if self.use_physics_prior:
                prior_map = self._generate_physics_prior(self.target_size[0], self.target_size[1], power)
            else:
                prior_map = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)

            for i in range(n_frames):
                # Map frame index 'i' to log index 'j'
                j = min(int(i * (n_logs / n_frames)), n_logs - 1)
                
                self.indices.append({
                    'video_path': v_path,
                    'frame_idx': i,
                    'temps': temps[j], # [T1, T2, T3, T4]
                    'sensor_pos': sensor_pos, # Dict of M1..M4
                    'prior_map': prior_map, # (1, H, W)
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
            # Return correct shapes: frame, target, mask, temps, prior, artifact_mask
            th, tw = self.target_size
            prior = item['prior_map'] if 'prior_map' in item else torch.zeros((1, th, tw), dtype=torch.float32)
            
            return (torch.zeros((3, th, tw)), 
                    torch.zeros((1, th, tw)), 
                    torch.zeros((1, th, tw)), 
                    torch.zeros((4,), dtype=torch.float32), 
                    prior, 
                    torch.zeros((1, th, tw)))

        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        h, w = item['original_size']
        th, tw = self.target_size
        
        frame_resized = cv2.resize(frame, (tw, th))
        
        # Transform Frame to Tensor
        frame_tensor = transforms.ToTensor()(frame_resized)
        frame_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                          std=[0.229, 0.224, 0.225])(frame_tensor)
        
        if self.transform:
            pass # TODO: apply extra transforms
        
        # --- Targets ---
        scale_x = tw / w
        scale_y = th / h
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
        artifact_mask = torch.zeros((1, th, tw), dtype=torch.float32)
        
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
            
            # Assign target
            val = float(temp_values[k])
            if not np.isnan(val):
                target_map[0, ny, nx] = val
                mask_map[0, ny, nx] = 1.0
            
            # Update Artifact Mask (region around sensor)
            if self.use_artifact_masking:
                radius = 4 # roughly 9x9 block for 256x256
                y_min = max(0, ny - radius)
                y_max = min(th, ny + radius + 1)
                x_min = max(0, nx - radius)
                x_max = min(tw, nx + radius + 1)
                artifact_mask[0, y_min:y_max, x_min:x_max] = 1.0
        
        if self.use_artifact_masking:
            # Detect Antenna Line
            # Use the resized frame for detection to match target size
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            edges = cv2.Canny(thresh, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=30, maxLineGap=10)
            
            antenna_mask = np.zeros((th, tw), dtype=np.float32)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # Draw a thick line on the mask
                    cv2.line(antenna_mask, (x1, y1), (x2, y2), 1.0, 10) 
            
            # Combine with sensor mask
            artifact_mask = torch.from_numpy(antenna_mask).unsqueeze(0) + artifact_mask
            artifact_mask = torch.clamp(artifact_mask, 0.0, 1.0)
            
            # Zero out pixels in frame where artifact_mask is 1.0
            # Need to broadcast (1, H, W) to (3, H, W)
            frame_tensor = frame_tensor * (1.0 - artifact_mask)

        # Prior Map
        prior = item['prior_map']
            
        return frame_tensor, target_map, mask_map, torch.tensor(temp_values, dtype=torch.float32), prior, artifact_mask

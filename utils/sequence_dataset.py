import os
import json
import glob
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SequenceHeatmapDataset(Dataset):
    def __init__(self, 
                 data_dir="data/level1_cropped", 
                 raw_dir="data/level0_raw", 
                 transform=None,
                 target_size=(64, 64),
                 sequence_length=16,
                 stride=8,
                 use_physics_prior=True):
        """
        Args:
            data_dir: Directory containing cropped MP4 videos and sensor_coordinates.json
            raw_dir: Directory containing CSV log files
            transform: Optional transform to be applied on the sequence frames.
            target_size: Tuple (H, W) per frame.
            sequence_length: Number of frames per sequence.
            stride: Step size between sequences (for sliding window).
            use_physics_prior: If True, returns Gaussian heatmap as prior.
        """
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.transform = transform
        self.target_size = target_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.use_physics_prior = use_physics_prior
        
        # Reuse logic to identify sensor map
        json_path = os.path.join(data_dir, "sensor_coordinates.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Sensor coordinates not found at {json_path}")
            
        with open(json_path, 'r') as f:
            self.coords_data = json.load(f)
            
        self.videos = [] # List of video metadata
        self.indices = [] # List of (video_idx, start_frame)
        
        self._index_dataset()
        
    @staticmethod
    def _parse_power(filename):
        try:
            parts = filename.split('_')
            for p in parts:
                if p.endswith('W'):
                    return float(p[:-1])
        except:
            pass
        return 30.0

    @staticmethod
    def _generate_physics_prior(H, W, power_watts):
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        sigma_x = 0.5 
        sigma_y = 0.5 
        
        dist = (xx**2)/(2*sigma_x**2) + (yy**2)/(2*sigma_y**2)
        heatmap = torch.exp(-dist)
        
        T_ambient = 37.0
        T_rise = power_watts * 0.6 
        
        prior = T_ambient + T_rise * heatmap
        return prior.unsqueeze(0) # (1, H, W)

    def _get_csv_path(self, identifier):
        # 1. Try 'cleaned' subdirectory first
        cleaned_pattern = os.path.join(self.raw_dir, "cleaned", f"LogJob_{identifier}_*.csv")
        candidates = glob.glob(cleaned_pattern)
        if candidates:
            return candidates[0]
            
        # 2. Try root
        search_pattern = os.path.join(self.raw_dir, f"*{identifier}*.csv")
        candidates = glob.glob(search_pattern)
        if candidates:
            log_jobs = [c for c in candidates if 'LogJob' in os.path.basename(c)]
            if log_jobs:
                return log_jobs[0]
            return candidates[0]
        return None

    def _index_dataset(self):
        video_files = sorted(glob.glob(os.path.join(self.data_dir, "*.mp4")))
        
        for v_idx, v_path in enumerate(video_files):
            fname = os.path.basename(v_path)
            
            parts = fname.split('_')
            try:
                # Expected format: US_001_30W_10min.mp4
                identifier = parts[1] 
                power = self._parse_power(fname)
            except:
                print(f"Skipping {fname}: filename format error")
                continue
                
            csv_path = self._get_csv_path(identifier)
            if not csv_path:
                print(f"Skipping {fname}: CSV not found for ID {identifier}")
                continue
                
            if fname not in self.coords_data:
                print(f"Skipping {fname}: No sensor coords for {fname}")
                continue
            sensor_pos = self.coords_data[fname]
            
            try:
                df = pd.read_csv(csv_path, sep=None, engine='python')
                req_cols = ['C26M1_Ch1', 'C26M2_Ch1', 'C26M3_Ch1', 'C26M4_Ch1']
                available_cols = [c for c in req_cols if c in df.columns]
                if len(available_cols) < 4:
                    print(f"Skipping {fname}: CSV missing columns")
                    continue
                temps_array = df[req_cols].values # (N_logs, 4)
            except Exception as e:
                print(f"Skipping {fname}: CSV error {e}")
                continue
                
            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened(): continue
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if self.use_physics_prior:
                prior_map = self._generate_physics_prior(
                    self.target_size[0], self.target_size[1], power
                )
            else:
                prior_map = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
                
            self.videos.append({
                'path': v_path,
                'temps': temps_array,
                'n_frames': n_frames,
                'n_logs': len(temps_array),
                'prior_map': prior_map,
                'sensor_pos': sensor_pos,
                'original_size': (height, width)
            })
            
            max_start = n_frames - self.sequence_length
            if max_start < 0:
                print(f"Skipping {fname}: Too short ({n_frames} frames) for seq_len {self.sequence_length}")
                continue
                
            for start_idx in range(0, max_start + 1, self.stride):
                self.indices.append((len(self.videos) - 1, start_idx))
                
        print(f"Indexed {len(self.videos)} videos, {len(self.indices)} sequences.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        video_idx, start_frame = self.indices[idx]
        meta = self.videos[video_idx]
        
        frames = []
        sparse_targets = []
        priors = []
        
        cap = cv2.VideoCapture(meta['path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        read_success = True
        
        for i in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                read_success = False
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            frame = frame.astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1) # (C, H, W)
            
            if self.transform:
                frame_tensor = self.transform(frame_tensor)
                
            frames.append(frame_tensor)
            
            global_frame_idx = start_frame + i
            log_idx = min(int(global_frame_idx * (meta['n_logs'] / meta['n_frames'])), meta['n_logs'] - 1)
            current_temps = meta['temps'][log_idx]
            
            sparse_map = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            
            orig_h, orig_w = meta['original_size']
            scale_y = self.target_size[0] / orig_h
            scale_x = self.target_size[1] / orig_w
            
            sensor_labels = ['M1', 'M2', 'M3', 'M4']
            for s_idx, label in enumerate(sensor_labels):
                if label not in meta['sensor_pos']: continue
                
                # Careful: The JSON keys might be "M1", "M2"... matching order
                center = meta['sensor_pos'][label]['center']
                sx, sy = center[0], center[1]
                
                px = int(sx * scale_x)
                py = int(sy * scale_y)
                px = min(max(px, 0), self.target_size[1]-1)
                py = min(max(py, 0), self.target_size[0]-1)
                
                temp_val = current_temps[s_idx]
                val = float(temp_val)
                if np.isnan(val): continue
                sparse_map[0, py, px] = val
                
            sparse_targets.append(sparse_map)
            priors.append(meta['prior_map'])
            
        cap.release()
        
        if not read_success or len(frames) != self.sequence_length:
            C, H, W = 3, self.target_size[0], self.target_size[1]
            return (torch.zeros((self.sequence_length, C, H, W)), 
                    torch.zeros((self.sequence_length, 1, H, W)), 
                    torch.zeros((self.sequence_length, 1, H, W)))

        frames_t = torch.stack(frames) # (Seq, C, H, W)
        targets_t = torch.stack(sparse_targets) # (Seq, 1, H, W)
        priors_t = torch.stack(priors) # (Seq, 1, H, W)
        
        return frames_t, targets_t, priors_t

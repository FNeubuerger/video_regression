import torch
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
import matplotlib.pyplot as plt
import numpy as np

def check_dataset_integrity():
    print("Checking dataset integrity...")
    
    # Initialize dataset
    dataset = TemperatureSequenceDataset(
        data_dir="data", 
        sequence_length=5, 
        image_size=(64, 64)
    )
    
    print(f"Total sequences: {len(dataset)}")
    
    # Check a few random sequences
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    for idx in indices:
        image_paths, temperatures = dataset.sequences[idx]
        print(f"\nSequence {idx}:")
        
        # Check frame numbers from paths
        frame_nums = []
        for p in image_paths:
            fname = os.path.basename(p)
            # Extract frame number
            import re
            match = re.match(r'frame_(\d+)_', fname)
            if match:
                frame_nums.append(int(match.group(1)))
        
        print(f"Frame numbers: {frame_nums}")
        print(f"Temperatures: {temperatures}")
        
        # Check if consecutive
        diffs = np.diff(frame_nums)
        print(f"Frame gaps: {diffs}")
        
        if np.any(diffs > 5):
            print("WARNING: Large gap detected!")
        if np.any(diffs <= 0):
            print("ERROR: Frames not sorted or duplicate!")

if __name__ == "__main__":
    check_dataset_integrity()

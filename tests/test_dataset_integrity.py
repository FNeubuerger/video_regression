import unittest
import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset

class TestDatasetIntegrity(unittest.TestCase):
    def test_dataloader_dims(self):
        """
        Iterates through the entire dataset with a batch size > 1 to ensure
        all items can be collated (i.e., have same dimensions).
        """
        data_dir = "data/level1_cropped"
        raw_dir = "data/level0_raw"
        
        # Ensure data directories exist
        if not os.path.exists(data_dir) or not os.path.exists(raw_dir):
            print(f"Data directories not found. Skipping test.")
            return

        # Initialize dataset with optical flow enabled, as that's the suspect
        dataset = SequenceHeatmapDataset(
            data_dir=data_dir,
            raw_dir=raw_dir,
            target_size=(64, 64),
            sequence_length=16,
            use_optical_flow=True,
            use_physics_prior=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Use batch_size=4 to force collation
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
        
        print("Iterating through DataLoader...")
        
        try:
            for i, batch in enumerate(tqdm(loader)):
                frames, targets, priors, scalars = batch
                
                # Check expected dimensions
                # Frames: (B, T, C, H, W). C should be 5 (3 RGB + 2 Flow)
                self.assertEqual(frames.shape[2], 5, f"Batch {i}: Expected 5 channels, got {frames.shape[2]}")
                self.assertEqual(frames.shape[1], 16, f"Batch {i}: Expected 16 frames, got {frames.shape[1]}")
                
        except RuntimeError as e:
            self.fail(f"DataLoader failed with RuntimeError: {e}")
        except Exception as e:
            self.fail(f"DataLoader failed with Exception: {e}")

if __name__ == "__main__":
    unittest.main()

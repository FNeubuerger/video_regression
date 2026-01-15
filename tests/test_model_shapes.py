import unittest
import torch
import torch.nn as nn
import os
import sys
from torchvision import transforms

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Models
from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from models.dense_heads import ResNetUNet
from models.bayesian import BayesianResNet, FullBayesianResNet
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from models.latent_ltc import LatentLTC_UNet
from models.conv_ltc import ConvLTC

# Import Dataset
from utils.sequence_dataset import SequenceHeatmapDataset

class TestArchitectureOnRealData(unittest.TestCase):
    
    def setUp(self):
        print("\n[SetUp] Loading one sample from SequenceHeatmapDataset...")
        self.data_dir = os.path.abspath("data/level1_cropped")
        
        # Prepare Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])
        
        try:
            self.dataset = SequenceHeatmapDataset(
                data_dir=self.data_dir,
                transform=transform,
                target_size=(64, 64),
                sequence_length=5,
                use_optical_flow=False # Start with 3 channels
            )
            
            # Fetch one item
            self.frames, self.targets, self.priors, self.scalars = self.dataset[0]
            
            # Add batch dimension -> (1, Seq, C, H, W)
            self.input_seq = self.frames.unsqueeze(0)
            
            # Single frame input (1, C, H, W) - take last frame
            self.input_frame = self.frames[-1].unsqueeze(0)
            
            self.seq_len, self.C, self.H, self.W = self.frames.shape
            self.frame_shape = (self.H, self.W, self.C) # (H, W, Channels)
            
            print(f"Data Loaded: Seq={self.input_seq.shape}")
            
        except Exception as e:
            self.fail(f"Failed to load dataset: {e}")

    # --- Standard Models ---
    def test_simple_resnet(self):
        """Test SimpleResNet (Scalar Regression)"""
        print("\nTesting SimpleResNet...")
        model = SimpleResNet(frame_shape=self.frame_shape)
        output = model(self.input_seq)
        
        # Output should be (Batch, 4) usually (4 sensors)
        print(f"SimpleResNet Output: {output.shape}")
        self.assertEqual(output.shape[0], 1)
        self.assertEqual(output.shape[1], 4)

    def test_cnnlstm(self):
        """Test CNNLSTM (Standard Time-Series)"""
        print("\nTesting CNNLSTM...")
        model = CNNLSTM(frame_shape=self.frame_shape, time_steps=self.seq_len)
        output = model(self.input_seq)
        
        print(f"CNNLSTM Output: {output.shape}")
        self.assertEqual(output.shape[0], 1)
        # Usually (B, 4) if it predicts for the sequence/latest
        # Or (B, 1)?
        # Implementation says: self.fc2 = nn.Linear(32, 4). So (B, 4).
        self.assertEqual(output.shape[1], 4)

    # --- Physics Models ---
    def test_physics_cnnlstm(self):
        """Test PhysicsCNNLSTM (Scalar Sequence)"""
        print("\nTesting PhysicsCNNLSTM...")
        model = PhysicsCNNLSTM(frame_shape=self.frame_shape, time_steps=self.seq_len, pretrained=False)
        output = model(self.input_seq)
        
        print(f"PhysicsCNNLSTM Output: {output.shape}")
        # Implementation returns (Batch, Time, 4)
        self.assertEqual(output.shape, (1, self.seq_len, 4))

    def test_spatial_physics_cnnlstm(self):
        """Test SpatialPhysicsCNNLSTM (Map Output)"""
        print("\nTesting SpatialPhysicsCNNLSTM...")
        model = SpatialPhysicsCNNLSTM(frame_shape=self.frame_shape, time_steps=self.seq_len, pretrained=False)
        temp_map, alpha, beta = model(self.input_seq)
        
        print(f"SpatialPhysics Output: Temp={temp_map.shape}, Alpha={alpha.shape}")
        # check temp_map: (B, T, 4, 4)
        self.assertEqual(temp_map.shape, (1, self.seq_len, 4, 4))
        # check alpha: (B, 1, 4, 4)
        self.assertEqual(alpha.shape, (1, 1, 4, 4))

    # --- Uncertainty Models ---
    def test_bayesian_resnet(self):
        """Test BayesianResNet"""
        print("\nTesting BayesianResNet...")
        model = BayesianResNet(frame_shape=self.frame_shape)
        output, kl = model(self.input_seq)
        
        print(f"BayesianResNet Output: {output.shape}, KL={kl.item()}")
        self.assertEqual(output.shape, (1, 4))
        self.assertTrue(isinstance(kl, torch.Tensor))

    # --- Dense Models (U-Net) ---
    def test_resnet_unet(self):
        """Test ResNetUNet (Dense Map)"""
        print("\nTesting ResNetUNet...")
        model = ResNetUNet(n_channels=self.C, n_classes=1)
        output = model(self.input_frame)
        print(f"ResNetUNet Output: {output.shape}")
        self.assertEqual(output.shape, (1, 1, self.H, self.W))

    # --- Dynamic Models (LTC) ---
    def test_latent_ltc(self):
        """Test LatentLTC_UNet"""
        print("\nTesting LatentLTC_UNet...")
        model = LatentLTC_UNet(n_channels=self.C, latent_dim=32, variational=False)
        
        # When variational=False, returns only prediction
        output = model(self.input_seq)
        
        print(f"LatentLTC Output: {output.shape}")
        self.assertEqual(output.shape, (1, self.seq_len, 1, self.H, self.W))
        
        # Verify variational mode
        print("Testing LatentLTC Variational Mode...")
        model_var = LatentLTC_UNet(n_channels=self.C, latent_dim=32, variational=True)
        output_var, kl_loss = model_var(self.input_seq)
        self.assertEqual(output_var.shape, (1, self.seq_len, 1, self.H, self.W))
        self.assertTrue(kl_loss is not None)

if __name__ == '__main__':
    unittest.main()

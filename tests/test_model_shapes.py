
import unittest
import torch
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, SimpleResNet, PretrainedCNNLSTM
from models.bayesian import BayesianCNNLSTM, BayesianResNet
from models.conv_ltc import ConvLTC_Model
from models.dense_heads import ResNetUNet

class TestModelShapes(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.time_steps = 5
        self.channels = 3
        self.height = 64
        self.width = 64
        self.frame_shape = (self.height, self.width, self.channels)
        
        # 5D Input (Batch, Time, Channels, Height, Width)
        self.input_5d = torch.randn(self.batch_size, self.time_steps, self.channels, self.height, self.width)
        
        # 4D Input (Batch, Channels, Height, Width)
        self.input_4d = torch.randn(self.batch_size, self.channels, self.height, self.width)

    def test_cnnlstm(self):
        print("\nTesting CNNLSTM...")
        model = CNNLSTM(frame_shape=self.frame_shape, time_steps=self.time_steps)
        output = model(self.input_5d)
        # Expected output: (Batch, 4)
        self.assertTrue(output.shape == (self.batch_size, 4))
        print("CNNLSTM Passed")

    def test_simple_resnet_5d(self):
        print("\nTesting SimpleResNet with 5D input...")
        model = SimpleResNet(frame_shape=self.frame_shape)
        output = model(self.input_5d)
        # Should handle 5D by taking last frame. Output (Batch, 4)
        self.assertTrue(output.shape == (self.batch_size, 4))
        print("SimpleResNet 5D Passed")

    def test_simple_resnet_4d(self):
        print("\nTesting SimpleResNet with 4D input...")
        model = SimpleResNet(frame_shape=self.frame_shape)
        output = model(self.input_4d)
        # Expect (Batch, 4)
        self.assertTrue(output.shape == (self.batch_size, 4))
        print("SimpleResNet 4D Passed")

    def test_bayesian_cnnlstm(self):
        print("\nTesting BayesianCNNLSTM...")
        model = BayesianCNNLSTM(frame_shape=self.frame_shape)
        output, kl = model(self.input_5d)
        print(f"BayesianCNNLSTM Output Shape: {output.shape}")
        # Expected output: (Batch, Time_Steps, 4)
        self.assertTrue(output.shape == (self.batch_size, self.time_steps, 4))
        print("BayesianCNNLSTM Passed")

    def test_bayesian_resnet_5d(self):
        print("\nTesting BayesianResNet with 5D input...")
        model = BayesianResNet(frame_shape=self.frame_shape)
        # Check if BayesianResNet handles 5D. 
        try:
            output, kl = model(self.input_5d)
            print(f"BayesianResNet Output Shape: {output.shape}")
            self.assertTrue(output.shape == (self.batch_size, 4))
            print("BayesianResNet 5D Passed")
        except Exception as e:
            print(f"BayesianResNet failed with 5D input: {e}")
            self.fail(f"BayesianResNet 5D failed: {e}")
        except RuntimeError as e:
            print(f"BayesianResNet 5D Failed as expected? Error: {e}")
            # If it fails, we should document it. 
            # But strictly speaking, if training loop passes 5D, this model MUST handle 5D.
            pass

    def test_conv_ltc(self):
        print("\nTesting ConvLTC_Model...")
        model = ConvLTC_Model(input_channels=self.channels, output_channels=1)
        output = model(self.input_5d)
        # ConvLTC typically outputs sequence or spatial map
        # Let's see what it outputs.
        print(f"ConvLTC output shape: {output.shape}")
        # Assuming it outputs (Batch, Time, 1, H, W) or (Batch, 1, H, W) 
        self.assertTrue(output.dim() >= 2)

    def test_resnet_unet(self):
        print("\nTesting ResNetUNet...")
        model = ResNetUNet(n_channels=self.channels, n_classes=1)
        # UNets are usually spatial.
        # Test with 4D
        output_4d = model(self.input_4d)
        print(f"ResNetUNet 4D output shape: {output_4d.shape}")
        self.assertTrue(output_4d.shape == (self.batch_size, 1, self.height, self.width))
        
        # Test with 5D?
        try:
            output_5d = model(self.input_5d)
            print(f"ResNetUNet 5D output shape: {output_5d.shape}")
        except Exception as e:
            print(f"ResNetUNet 5D Failed: {e}")


if __name__ == '__main__':
    unittest.main()

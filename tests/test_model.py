import pytest
import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dense_heads import ResNetUNet
from models.backbones import SimpleResNet
from models.bayesian import BayesianResNet

def test_resnet_unet_structure():
    # Pass n_channels=3 to match the input tensor
    model = ResNetUNet(n_channels=3, n_classes=1)
    # Basic input: Batch=2, Channels=3, H=64, W=64
    x = torch.randn(2, 3, 64, 64)
    # Forward pass
    out = model(x)
    
    # Check output shape: Batch=2, Channels=1, H=64, W=64
    assert out.shape == (2, 1, 64, 64)

def test_simple_resnet_structure():
    model = SimpleResNet(frame_shape=(64, 64, 3))
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    # Model regresses the 4 sensor temperatures per sample, shape (Batch, 4)
    assert out.shape == (2, 4)

def test_bayesian_resnet_structure():
    model = BayesianResNet(frame_shape=(64, 64, 3))
    x = torch.randn(2, 3, 64, 64)
    # bnn forward returns (prediction, kl_divergence)
    out, kl = model(x)
    assert out.shape == (2, 4)
    assert kl.numel() == 1


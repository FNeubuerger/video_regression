import pytest
import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dense_heads import ResNetUNet
from models.backbones import SimpleResNet, CNNLSTM
from models.bayesian import BayesianResNet
from models.latent_ltc import LatentLTC_UNet
from models.conv_ltc import ConvLTC_Model

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

def test_cnnlstm_structure():
    # CNNLSTM expects (Batch, Time, Channels, H, W)
    time_steps = 5
    model = CNNLSTM(frame_shape=(64, 64, 3), time_steps=time_steps)
    x = torch.randn(2, time_steps, 3, 64, 64)
    out = model(x)
    # Model regresses the 4 sensor temperatures per sample, shape (Batch, 4)
    assert out.shape == (2, 4)

def test_latent_ltc_unet_structure():
    # Latent LTC Expects (Batch, Time, Channels, H, W)
    # Returns (Batch, Time, 1, H, W) - Dense map per timestep
    batch_size = 2
    seq_len = 4
    ncp_units = 32
    latent_dim = 16 # small for test
    # Ensure ncp_units > latent_dim for AutoNCP
    ncp_units = 50 
    
    model = LatentLTC_UNet(n_channels=3, latent_dim=latent_dim, ncp_units=ncp_units)
    x = torch.randn(batch_size, seq_len, 3, 64, 64)
    
    out = model(x)
    assert out.shape == (batch_size, seq_len, 1, 64, 64)


def test_conv_ltc_structure():
    # ConvLTC Expects (Batch, Time, Channels, H, W)
    # Returns (Batch, Time, 1, H, W)
    batch_size = 2
    seq_len = 3
    # Use small spatial dim for speed
    H, W = 16, 16 
    
    model = ConvLTC_Model(input_channels=3, hidden_channels=8, output_channels=1)
    x = torch.randn(batch_size, seq_len, 3, H, W)
    
    out = model(x)
    assert out.shape == (batch_size, seq_len, 1, H, W)

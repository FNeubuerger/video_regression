
import pytest
import torch
import torch.nn as nn
from models.dense_heads import ResNetUNet
from models.latent_ltc import LatentLTC_UNet

# Use a fixture for device to allow easy switching if needed
@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestBayesianResNetUNet:
    def test_variational_output_structure(self, device):
        """Test that variational=True returns (tensor, kl_loss)"""
        model = ResNetUNet(n_channels=3, n_classes=1, variational=True).to(device)
        batch_size = 2
        x = torch.randn(batch_size, 3, 64, 64).to(device)
        
        output = model(x)
        
        # Should return a tuple
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        pred, kl = output
        
        # Check prediction shape
        assert pred.shape == (batch_size, 1, 64, 64)
        
        # Check KL is a scalar tensor
        assert isinstance(kl, torch.Tensor)
        assert kl.ndim == 0
        assert kl.item() >= 0 # KL divergence should be non-negative

    def test_deterministic_output_structure(self, device):
        """Regression test for variational=False"""
        model = ResNetUNet(n_channels=3, n_classes=1, variational=False).to(device)
        x = torch.randn(2, 3, 64, 64).to(device)
        
        output = model(x)
        
        # Should return just the tensor
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 1, 64, 64)

class TestBayesianLatentLTC:
    def test_variational_output_structure(self, device):
        """Test that variational=True returns (tensor, kl_loss)"""
        # LatentLTC outputs are hardcoded to decode to 64x64 currently in the model definition
        model = LatentLTC_UNet(n_channels=3, latent_dim=16, ncp_units=20, variational=True).to(device)
        
        batch_size = 2
        seq_len = 5
        # The encoder uses ResNet18, which reduces spatial dims by 32x. 
        # 64x64 input -> 2x2 feature map.
        x = torch.randn(batch_size, seq_len, 3, 64, 64).to(device)
        
        output = model(x)
        
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        pred, kl = output
        
        # Model decoder outputs 64x64
        assert pred.shape == (batch_size, seq_len, 1, 64, 64)
        
        assert isinstance(kl, torch.Tensor)
        assert kl.ndim == 0

    def test_deterministic_output_structure(self, device):
        """Regression test for variational=False"""
        model = LatentLTC_UNet(n_channels=3, latent_dim=16, ncp_units=20, variational=False).to(device)
        batch_size = 2
        seq_len = 3
        x = torch.randn(batch_size, seq_len, 3, 64, 64).to(device)
        
        # Should return just the prediction tensor
        # Note: The original LatentLTC definition might iterate and return valid sequences.
        # Based on previous read, it returns `pred_map` directly.
        
        output = model(x)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (batch_size, seq_len, 1, 64, 64)

if __name__ == "__main__":
    pytest.main([__file__])

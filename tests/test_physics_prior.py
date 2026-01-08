import pytest
import torch
import sys
import os
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset

def test_prior_logic_standalone():
    """
    Test the prior generation logic without loading files.
    """
    # Use static methods directly on the class
    
    # Check 30W
    prior_30 = TemperatureHeatmapDataset._generate_physics_prior(64, 64, 30.0)
    assert prior_30.shape == (1, 64, 64)
    max_30 = prior_30.max()
    
    # Check 50W
    prior_50 = TemperatureHeatmapDataset._generate_physics_prior(64, 64, 50.0)
    max_50 = prior_50.max()
    
    # Higher wattage should mean higher peak temperature
    assert max_50 > max_30
    
    # Check center is hottest
    center_val = prior_30[0, 32, 32]
    edge_val = prior_30[0, 0, 0]
    assert center_val > edge_val

def test_prior_integration_with_random_tensors():
    """
    Simulate what happens inside the training loop.
    Delta + Prior = Prediction
    """
    batch_size = 4
    h, w = 64, 64
    
    # Mock model output (Delta)
    delta_pred = torch.randn(batch_size, 1, h, w)
    
    # Mock Priors
    priors = torch.ones(batch_size, 1, h, w) * 37.0 # Body temp
    
    # Final Prediction
    final_pred = delta_pred + priors
    
    assert final_pred.shape == (batch_size, 1, h, w)
    assert torch.allclose(final_pred - delta_pred, priors)

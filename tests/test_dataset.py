import pytest
import torch
import sys
import os
import shutil

# Add parent to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset

@pytest.fixture
def mock_dataset_structure(tmp_path):
    """
    Creates a temporary directory structure mimicking the real dataset.
    """
    data_dir = tmp_path / "data" / "level1_cropped"
    raw_dir = tmp_path / "data" / "level0_raw"
    data_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    
    # Create dummy video
    # We can't easily create a valid mp4 without opencv/ffmpeg, 
    # so we might need to mock cv2.VideoCapture or just skip the file reading part in integration tests.
    # For now, let's just assume the file exists and mock the loader if possible,
    # OR create a very small valid mp4 if we want end-to-end.
    # Easier: Mock the _index_dataset logic or use mocking.
    
    # Let's rely on the actual code being robust enough or use `unittest.mock`
    pass
    return data_dir, raw_dir

# For now, since we have the data on disk, we can test with a subset of the actual data
# This is an "Integration Test" more than a Unit Test, but valuable.
DATA_EXISTS = os.path.exists("data/level1_cropped") and os.path.exists("data/level0_raw")

@pytest.mark.skipif(not DATA_EXISTS, reason="Real data not found")
def test_dataset_loading():
    ds = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64)
    )
    
    assert len(ds) > 0
    
    # Get a sample
    sample = ds[0]
    frame, target, mask, temp_vec, prior = sample
    
    # Check shapes
    assert frame.shape == (3, 64, 64)
    assert target.shape == (1, 64, 64)
    assert mask.shape == (1, 64, 64)
    assert temp_vec.shape == (4,)
    assert prior.shape == (1, 64, 64)

@pytest.mark.skipif(not DATA_EXISTS, reason="Real data not found")
def test_dataset_prior_generation():
    ds = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64)
    )
    
    # Test valid wattage parsing
    power = ds._parse_power_from_filename("US_005_30W_10min.mp4")
    assert power == 30.0
    
    # Test unknown parsing (fallback)
    power_unk = ds._parse_power_from_filename("Random_Video.mp4")
    assert power_unk == 30.0 # Current fallback
    
    # Test Grid Generation
    prior = ds._generate_physics_prior(32, 32, 50.0)
    assert prior.shape == (1, 32, 32)
    assert prior.max() > prior.min() # Should have variance


import torch
from torch.utils.data import DataLoader
from utils.sequence_dataset import SequenceHeatmapDataset
from models.backbones import SimpleResNet
import os

def test_simpleresnet_shape():
    # Setup paths
    data_dir = "data/level1_cropped"
    dataset = SequenceHeatmapDataset(
        data_dir=data_dir,
        sequence_length=10, # Standard length
        target_size=(64, 64),
        transform=None
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Get one batch
    batch = next(iter(loader))
    if len(batch) == 3:
        images, labels, mask = batch
    else:
        images, labels = batch
        
    print(f"Batch shapes: Images={images.shape}, Labels={labels.shape}")
    
    # Init model
    model = SimpleResNet(frame_shape=(64, 64, 3))
    
    # Run forward
    try:
        output = model(images)
        print(f"Model output shape: {output.shape}")
        print("SimpleResNet forward pass successful!")
    except Exception as e:
        print(f"SimpleResNet failed: {e}")
        # Print input details if fail
        print(f"Input dim: {images.dim()}")

if __name__ == "__main__":
    test_simpleresnet_shape()

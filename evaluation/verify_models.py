
import torch
import sys
import os
import json
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# Add parent directory to path
sys.path.append(os.path.abspath('.'))

from utils.sequence_dataset import SequenceHeatmapDataset
from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from physics import PhysicsCNNLSTM 
from physics.models import SpatialPhysicsCNNLSTM

def create_model_verification(model_type, frame_shape, time_steps):
    """Replicates create_model from train_all_models.py"""
    if model_type == "cnnlstm":
        return CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "pretrained_cnnlstm":
        pretrained_cnn = resnet18(weights='IMAGENET1K_V1')
        pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)
        return PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "simple_resnet":
        return SimpleResNet(frame_shape=frame_shape)
    elif model_type == "physics_cnnlstm":
        return PhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
    elif model_type == "spatial_physics_cnnlstm":
        return SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def verify_all_models():
    print("=== Model Verification Started ===")
    
    # 1. Setup Data
    print("Step 1: Loading Dataset...")
    data_dir = "data/level1_cropped"
    dataset = SequenceHeatmapDataset(
        data_dir=data_dir,
        sequence_length=10, 
        target_size=(64, 64),
        transform=None
    )
    
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("Error: Dataset is empty.")
        return

    if len(batch) == 3:
        images, labels, mask = batch
    else:
        images, labels = batch
        mask = None
        
    print(f"Data Shapes - Images: {images.shape}, Labels: {labels.shape}")
    if mask is not None:
        print(f"Mask Shape: {mask.shape}")
        
    # Check if images are 5D (B, T, C, H, W)
    if images.dim() != 5:
        print(f"WARNING: Expected 5D input (B, T, C, H, W), got {images.dim()}D")
    
    # 2. Iterate Models
    model_types = [
        "cnnlstm",
        "pretrained_cnnlstm", 
        "simple_resnet",
        "physics_cnnlstm",
        "spatial_physics_cnnlstm"
    ]
    
    frame_shape = (64, 64, 3) # (H, W, C) - dataset returns channels first though?
    # PyTorch uses (C, H, W). The model init expects 'frame_shape'.
    # Looking at backbones.py: 
    # frame_shape[2] is accessed for channels. 
    # So it expects (H, W, C) or (C, H, W)?
    # line 41 in backbones.py: nn.Conv2d(frame_shape[2], 16, ...)
    # If frame_shape is (64, 64, 3), then frame_shape[2] is 3. Correct.
    # Dataset returns (B, T, 3, 64, 64).
    
    time_steps = 10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    images = images.to(device)
    
    for m_type in model_types:
        print(f"\n--- Verifying {m_type} ---")
        try:
            model = create_model_verification(m_type, frame_shape, time_steps)
            model = model.to(device)
            
            output = model(images)
            
            print(f"Forward Pass: SUCCESS")
            if isinstance(output, tuple):
                 print(f"Output Shape: Tuple of length {len(output)}")
                 print(f"Main Output: {output[0].shape}")
            else:
                 print(f"Output Shape: {output.shape}")
                 
        except Exception as e:
            print(f"Msg: {e}")
            print(f"FAIL: {m_type} crashed.")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    verify_all_models()

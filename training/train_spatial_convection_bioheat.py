"""
Training script for Spatial Convection Bioheat (Single Frame).
Includes Optical Flow to model convection: v . grad T.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset
from models.backbones import SpatialResNet
from physics.bioheat_loss import AdvancedBioHeatLoss

def train_spatial_convection_bioheat(epochs=50, batch_size=32, learning_rate=1e-4):
    wandb.init(project="video-temperature-regression", name="spatial-convection-bioheat")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 1 
    
    print(f"Training Spatial Convection Bioheat on {device}")
    
    # Data
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
        sequence_length=sequence_length, 
        target_size=(64, 64), 
        use_optical_flow=True # Needed for convection
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    # Input channels = 5 (RGB + Flow) or just RGB?
    # SpatialResNet usually takes RGB.
    # But convection requires Flow.
    # Option 1: Pass RGB to model, Flow to Loss.
    # Option 2: Pass RGB+Flow to model.
    # Logic: Model predicts T using Flow info? Yes, helpful.
    print("Initializing SpatialResNet with 5 channels...")
    model = SpatialResNet(frame_shape=(64, 64, 5))
    model.to(device)
    
    # Loss
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01,
        initial_conductivity=0.001,
        learnable_params=True,
        spatial_params=False
    ).to(device)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(criterion.parameters()), 
        lr=learning_rate
    )
    
    # Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for frames, maps, priors, scalars in progress:
            # frames: (B, 1, 5, H, W) -> (B, 5, H, W)
            frames = frames.squeeze(1).to(device)
            # targets: (B, 1, H, W) -> (B, H, W)
            maps = maps.squeeze(1).to(device).float()
            
            # Extract Flow for Loss (Channels 3, 4)
            # Flow shape for Loss: (B, 2, H, W) (Single frame)
            # frames is (B, 5, H, W).
            flow = frames[:, 3:, :, :] 
            
            optimizer.zero_grad()
            
            # Model output: (B, 1, H, W)
            # Note: SpatialResNet output usually (B, H, W) or (B, 1, H, W)?
            # Needs checking. Assuming (B, 1, H, W).
            predictions = model(frames)
            
            loss = criterion(predictions, maps, flow=flow)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, maps, priors, scalars in val_loader:
                frames = frames.squeeze(1).to(device)
                maps = maps.squeeze(1).to(device).float()
                flow = frames[:, 3:, :, :]
                
                predictions = model(frames)
                loss = criterion(predictions, maps, flow=flow)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/spatial_convection_bioheat_resnet.pth")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_spatial_convection_bioheat(epochs=args.epochs)

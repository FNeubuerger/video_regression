"""
Training script for Spatial Bioheat-Informed ResNet (Single Frame).
Uses AdvancedBioHeatLoss in spatial-only mode (Laplacian regularization).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tqdm import tqdm
import wandb

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset
from models.backbones import SpatialResNet
from physics.bioheat_loss import AdvancedBioHeatLoss

def train_spatial_bioheat_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="spatial-bioheat-resnet")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We use sequence_length=1 for single frame training, or load 5 and flatten?
    # Better to load 1 for efficiency if Dataset supports it.
    sequence_length = 1 
    
    print(f"Training Spatial Bioheat Model (Single Frame) on {device}")
    
    # Data
    print("Loading dataset...")
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
        sequence_length=sequence_length, 
        target_size=(64, 64),
        use_optical_flow=False 
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Initializing SpatialResNet...")
    model = SpatialResNet(frame_shape=(64, 64, 3))
    model.to(device)
    
    # Advanced Bioheat Loss
    # dt parameter is ignored for single frame (or used as 1.0)
    # spatial_params=True implies we learn alpha/beta maps??
    # Issue #41 says "Spatial Parameter Discovery".
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        lr_params=True,
        spatial_params=False # Start simple scalar params first
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
            # frames: (B, 1, C, H, W)
            frames = frames.squeeze(1).to(device) # (B, C, H, W)
            
            # targets: (B, 1, 4) -> (B, 4)
            scalars = scalars.squeeze(1).to(device).float()
            
            # maps: (B, 1, 1, H, W) -> (B, 1, H, W)
            maps = maps.squeeze(1).to(device).float()

            optimizer.zero_grad()
            
            # Prediction: (B, 1, H, W) - Map output from SpatialResNet?
            # Or does SpatialResNet output map?
            # Check models/backbones.py (I confirmed it exists but didn't read forward)
            # Assuming it outputs a map.
            predictions = model(frames)
            
            # Loss
            # Compare Map Prediction vs Scalar Targets?
            # AdvancedBioHeatLoss handles (B, H, W) vs (B, 4) by GAP if needed?
            # Or compare Map vs Map?
            # Usually Spatial Bioheat comparison is Map vs Dense Map.
            # Using 'maps' (Dense Target)
            loss = criterion(predictions, maps)
            
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
                
                predictions = model(frames)
                loss = criterion(predictions, maps)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch + 1
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/spatial_bioheat_resnet.pth")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_spatial_bioheat_model(epochs=args.epochs)

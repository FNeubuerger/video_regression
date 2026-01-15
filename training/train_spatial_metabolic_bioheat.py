"""
Training script for Spatial Metabolic Bioheat (Single Frame).
Includes Metabolic Rate (Q_met) to model heat generation.
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

def train_spatial_metabolic_bioheat(epochs=50, batch_size=32, learning_rate=1e-4):
    wandb.init(project="video-temperature-regression", name="spatial-metabolic-bioheat")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 1 
    
    print(f"Training Spatial Metabolic Bioheat on {device}")
    
    # Data
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
        sequence_length=sequence_length, 
        target_size=(64, 64),
        use_optical_flow=False 
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Initializing SpatialResNet...")
    model = SpatialResNet(frame_shape=(64, 64, 3))
    model.to(device)
    
    # Loss
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01,
        initial_metabolic_rate=1000.0,
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
            frames = frames.squeeze(1).to(device)
            maps = maps.squeeze(1).to(device).float()
            
            optimizer.zero_grad()
            predictions = model(frames)
            
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
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/spatial_metabolic_bioheat_resnet.pth")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_spatial_metabolic_bioheat(epochs=args.epochs)

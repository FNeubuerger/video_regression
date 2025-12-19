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

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
from models.backbones import SpatialResNet
from physics.bioheat_loss import AdvancedBioHeatLoss
from tqdm import tqdm
import wandb

def train_spatial_bioheat_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="spatial-metabolic-bioheat-resnet")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # We use sequence_length=1 for single frame training
    sequence_length = 1 
    
    print(f"Training Spatial Metabolic Bioheat Model (Single Frame) on {device}")
    
    # Data
    print("Loading dataset...")
    dataset = TemperatureSequenceDataset(
        data_dir="data", 
        sequence_length=sequence_length, 
        image_size=(64, 64),
        use_optical_flow=True 
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Initializing SpatialResNet...")
    # Input: 5 channels (3 RGB + 2 Flow)
    model = SpatialResNet(frame_shape=(64, 64, 5), output_map_size=(4, 4))
    model.to(device)
    
    # Advanced Bioheat Loss
    # Learnable parameters enabled
    # Note: Since we only have single frames, the loss will only enforce spatial smoothness (Laplacian)
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01, 
        initial_conductivity=0.001,
        initial_metabolic_rate=0.005,
        arterial_temp=37.0,
        learnable_params=True,
        dx=1.0 # mm
    ).to(device)
    
    # Optimizer includes model params AND loss params
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
        for images, labels in progress:
            images = images.to(device) # (B, 1, 5, 64, 64)
            labels = labels.to(device).float() # (B, 1)
            
            # Squeeze time dimension for single frame model
            images = images.squeeze(1) # (B, 5, 64, 64)
            # labels is already (B), no need to squeeze
            
            optimizer.zero_grad()
            
            # Predictions are (B, 4, 4) maps
            predictions = model(images)
            
            # Add dummy time dimension for compatibility with loss
            predictions_unsqueezed = predictions.unsqueeze(1) # (B, 1, 4, 4)
            labels_unsqueezed = labels.unsqueeze(1) # (B, 1)
            
            # Extract flow
            raw_flow = images[:, 3:5, :, :] # (B, 2, 64, 64)
            
            # Downsample flow
            flow_downsampled = torch.nn.functional.interpolate(
                raw_flow, size=(4, 4), mode='area'
            )
            flow_unsqueezed = flow_downsampled.unsqueeze(1) # (B, 1, 2, 4, 4)
            
            # Loss returns total_loss, alpha, beta
            loss, alpha, beta = criterion(predictions_unsqueezed, labels_unsqueezed, flow=flow_unsqueezed)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress.set_postfix(loss=loss.item(), alpha=f"{alpha:.4f}", beta=f"{beta:.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device).squeeze(1)
                labels = labels.to(device).float().squeeze(1)
                
                predictions = model(images)
                loss, _, _ = criterion(predictions, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        wandb.log({
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "alpha": criterion.alpha.item(),
            "beta": criterion.beta.item()
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/spatial_metabolic_bioheat_resnet.pth")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_spatial_bioheat_model(epochs=args.epochs)

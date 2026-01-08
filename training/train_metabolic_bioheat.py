"""
Training script for Bioheat-Informed CNN-LSTM.
Uses BioHeatEquationLoss to enforce Pennes' Bioheat Equation constraints.
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
from physics.models import SpatialPhysicsCNNLSTM
from physics.bioheat_loss import AdvancedBioHeatLoss
from tqdm import tqdm
import wandb

def train_bioheat_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="metabolic-bioheat-cnnlstm")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5 
    
    print(f"Training Metabolic Bioheat Model on {device}")
    
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
    print("Initializing SpatialPhysicsCNNLSTM...")
    model = SpatialPhysicsCNNLSTM(frame_shape=(64, 64, 5), time_steps=sequence_length, pretrained=True)
    model.to(device)
    
    # Advanced Bioheat Loss
    # Learnable parameters enabled
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01, 
        initial_conductivity=0.001,
        initial_metabolic_rate=0.005, # Initialize with some metabolic activity
        arterial_temp=37.0,
        learnable_params=True,
        dx=1.0 # mm
    ).to(device) # Parameters need to be on device
    
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
            images = images.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            
            # Predictions are now (B, T, 4, 4) maps
            predictions = model(images)
            
            # Extract flow from images (B, T, 5, H, W) -> (B, T, 2, H, W)
            # Channels 3 and 4 are flow
            raw_flow = images[:, :, 3:5, :, :]
            
            # Downsample flow to match prediction size (4x4)
            B, T, C_flow, H_flow, W_flow = raw_flow.shape
            raw_flow_reshaped = raw_flow.view(B * T, C_flow, H_flow, W_flow)
            
            # Use area interpolation for downsampling flow (averaging vectors)
            flow_downsampled = torch.nn.functional.interpolate(
                raw_flow_reshaped, size=(4, 4), mode='area'
            )
            
            flow = flow_downsampled.view(B, T, 2, 4, 4)
            
            # Loss returns total_loss, alpha, beta
            loss, alpha, beta = criterion(predictions, labels, flow=flow)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress.set_postfix(loss=loss.item(), alpha=f"{alpha:.4f}", beta=f"{beta:.4f}", qm=f"{criterion.qm.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                predictions = model(images)
                loss, _, _ = criterion(predictions, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        print(f"Physics Params: Alpha={criterion.alpha.item():.5f}, Beta={criterion.beta.item():.5f}, Qm={criterion.qm.item():.5f}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "alpha": criterion.alpha.item(),
            "beta": criterion.beta.item(),
            "qm": criterion.qm.item(),
            "epoch": epoch + 1
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/metabolic_bioheat_model.pth")
            print("Saved best model.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_bioheat_model(epochs=args.epochs)

"""
Training script for Bioheat-Informed CNN-LSTM (Scalar).
Uses BioHeatEquationLoss to enforce Pennes' Bioheat Equation constraints on the temporal prediction.
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
from physics.models import PhysicsCNNLSTM
from physics.bioheat_loss import AdvancedBioHeatLoss

def train_bioheat_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="bioheat-cnnlstm-scalar")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5 
    
    print(f"Training Bioheat PINN (Scalar) on {device}")
    
    # Data
    print("Loading dataset...")
    # Using 'level1_cropped' as per new plan
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
    
    # Model: PhysicsCNNLSTM (Scalar Sequence Output)
    print("Initializing PhysicsCNNLSTM...")
    # Frame shape is (H, W, C) -> (64, 64, 3)
    model = PhysicsCNNLSTM(frame_shape=(64, 64, 3), time_steps=sequence_length, pretrained=True)
    model.to(device)
    
    # Advanced Bioheat Loss
    # We use dt=1.0 assumption for now, or match dataset FPS
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01, 
        arterial_temp=37.0,
        learnable_params=True,
        dt=1.0 
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
        # Dataset returns: frames, targets(map), priors, scalars
        for frames, maps, priors, scalars in progress:
            frames = frames.to(device)
            scalars = scalars.to(device).float() # (B, T, 4)
            
            optimizer.zero_grad()
            
            # Predictions: (B, T, 4)
            predictions = model(frames)
            
            # Use scalars as target. 
            # Note: The Physics Loss will enforce smoothness on the sequence.
            loss = criterion(predictions, scalars)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress.set_postfix(loss=loss.item(), alpha=f"{criterion.alpha.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, maps, priors, scalars in val_loader:
                frames = frames.to(device)
                scalars = scalars.to(device).float()
                predictions = model(frames)
                loss = criterion(predictions, scalars)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "alpha": criterion.alpha.item(),
            "epoch": epoch + 1
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/bioheat_pinn_model.pth")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_bioheat_model(epochs=args.epochs)

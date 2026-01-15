"""
Training script for Bayesian Physics-Informed Neural Network (B-PINN).
Combines Bayesian Neural Networks (Uncertainty) with Bioheat Loss (Physics).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
import argparse
from tqdm import tqdm
import wandb
import torchbnn as bnn

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.sequence_dataset import SequenceHeatmapDataset
from models.bayesian import BayesianResNet, BayesianCNNLSTM
from physics.bioheat_loss import AdvancedBioHeatLoss

def train_bayesian_pinn(epochs=50, batch_size=32, learning_rate=1e-4, kl_weight=0.1):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="bayesian-cnnlstm")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5 
    
    print(f"Training Bayesian PINN on {device}")
    
    # Data
    print("Loading dataset...")
    # Use the new dataset class
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
        sequence_length=sequence_length, 
        target_size=(64, 64),
        use_optical_flow=True,
        use_physics_prior=False
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    # Use BayesianCNNLSTM for temporal processing
    # Frame shape is (H, W, C) -> (64, 64, 5) with flow
    print("Initializing BayesianCNNLSTM...")
    model = BayesianCNNLSTM(frame_shape=(64, 64, 5))
    model.to(device)
    
    # Criterion: Advanced Bioheat Loss
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        learnable_params=True
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
        train_kl = 0.0
        train_phys = 0.0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for frames, maps, priors, scalars in progress:
            frames = frames.to(device) # (B, T, 5, 64, 64)
            
            # Target (Scalars)
            scalars = scalars.to(device).float() # (B, T, 4)
            
            # Flow extraction (channels 3, 4)
            flow = frames[:, :, 3:, :, :] # (B, T, 2, H, W)
            
            optimizer.zero_grad()
            
            # Bayesian Forward Pass
            # Returns (predictions, kl_loss)
            predictions, kl_loss = model(frames) # predictions: (B, T, 4)
            
            # 1. Physics/MSE Loss
            # AdvancedBioHeatLoss expects specific shapes.
            # predictions: (B, T, 4)
            # targets: (B, T, 4)
            # flow: (B, T, 2, H, W)
            # Note: The loss likely averages over sensors (scalar) unless map is provided?
            # AdvancedBioHeatLoss calculates diffusion on MAPS.
            # If we pass SCALARS, it skips diffusion/convection terms if dim < 3 or 4.
            # BUT we want PINN constraints.
            # For Scalar PINN, we rely on temporal derivative dT/dt + analytical source/perfusion?
            # Or is this model supposed to predict MAPS?
            # BayesianCNNLSTM usually predicts SCALARS.
            # If so, Bioheat Loss is just ODE (dT/dt = P - D) without spatial laplacian.
            # This is valid.
            
            phys_loss = criterion(
                predictions=predictions, 
                targets=scalars, 
                flow=None, # Cannot use flow for scalar equation convection (no gradient T_x)
                mask=None
            )
            
            # Total Loss = Physics/MSE Loss + KL Weight * KL Loss
            loss = phys_loss + kl_weight * kl_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_kl += kl_loss.item()
            train_phys += phys_loss.item()
            
            progress.set_postfix(
                loss=loss.item(), 
                kl=f"{kl_loss.item():.2f}", 
                phys=f"{phys_loss.item():.2f}"
            )
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for frames, maps, priors, scalars in val_loader:
                frames = frames.to(device)
                scalars = scalars.to(device).float()
                
                predictions, kl_loss = model(frames)
                phys_loss = criterion(predictions, scalars)
                loss = phys_loss + kl_weight * kl_loss
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "kl_loss": train_kl / len(train_loader),
            "phys_loss": train_phys / len(train_loader),
            "epoch": epoch + 1
        })
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/bayesian_cnnlstm.pth")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_bayesian_pinn(epochs=args.epochs)

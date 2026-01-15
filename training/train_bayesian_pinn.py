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

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
from utils.sequence_dataset import SequenceHeatmapDataset
from models.bayesian import BayesianResNet
from physics.bioheat_loss import AdvancedBioHeatLoss
from tqdm import tqdm
import wandb
import torchbnn as bnn

def train_bayesian_pinn(epochs=50, batch_size=32, learning_rate=1e-4, kl_weight=0.1, masked=False):
    # Initialize WandB
    run_name = "bayesian-pinn"
    if masked:
        run_name += "-masked"
    wandb.init(project="video-temperature-regression", name=run_name)
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5
    
    # Data
    print("Loading dataset...")
    # Use SequenceHeatmapDataset to get access to 4-channel scalar targets
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
        raw_dir="data/level0_raw",
        sequence_length=sequence_length, 
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=masked
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Initializing BayesianResNet...")
    # Input: 5 channels (3 RGB + 2 Flow)
    model = BayesianResNet(frame_shape=(64, 64, 5)).to(device)
    
    # Advanced Bioheat Loss
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01, 
        initial_conductivity=0.001,
        arterial_temp=37.0,
        learnable_params=True,
        dt=1.0
    ).to(device)
    
    mse_criterion = nn.MSELoss()
    kl_loss_fn = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    
    # Optimizer
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
        for batch in progress:
            # SequenceHeatmapDataset returns: frames_t, targets_t, priors_t, scalars_t
            # frames: (B, T, 5, H, W)
            # targets: (B, T, 1, H, W) (sparse map)
            # priors: (B, T, 1, H, W)
            # scalars: (B, T, 4) (all sensors)
            
            if len(batch) == 4:
                images, _, _, labels = batch
                mask = None
            else:
                # Fallback or strict handling
                raise ValueError(f"Unexpected batch length: {len(batch)}")
            
            images = images.to(device) # (B, T, 5, 64, 64)
            labels = labels.to(device).float() # (B, T, 4)
            
            optimizer.zero_grad()
            
            # Bayesian Forward Pass
            # We need predictions for the whole sequence to compute dT/dt
            # BayesianResNet takes (B, C, H, W) usually.
            # We reshape: (B*T, C, H, W)
            B, T, C, H, W = images.shape
            images_flat = images.view(B*T, C, H, W)
            
            # Forward pass (Monte Carlo sampling happens implicitly if we just do one pass, 
            # but for training we usually just do one pass per batch and rely on stochasticity)
            preds_flat, kl_div = model(images_flat) # (B*T, 4)
            
            # No averaging needed, we want (B, T, 4)
            predictions = preds_flat.view(B, T, 4)
            
            # 1. Physics Loss (includes MSE + Physics)
            # This returns the combined loss
            phys_loss, alpha, beta = criterion(predictions, labels)
            
            # 2. KL Divergence Loss (Bayesian Regularization)
            kl = kl_div
            
            # Total Loss
            total_loss = phys_loss + (kl_weight * kl)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_kl += kl.item()
            train_phys += phys_loss.item()
            
            progress.set_postfix(loss=total_loss.item(), kl=kl.item(), phys=phys_loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    images, _, _, labels = batch
                    mask = None
                else:
                    raise ValueError(f"Unexpected batch length: {len(batch)}")
                    
                images = images.to(device)
                labels = labels.to(device).float()
                    
                B, T, C, H, W = images.shape
                images_flat = images.view(B*T, C, H, W)
                preds_flat, _ = model(images_flat)
                
                # No averaging
                predictions = preds_flat.view(B, T, 4)
                
                phys_loss, _, _ = criterion(predictions, labels)
                # KL is usually not computed on validation or is constant
                val_loss += phys_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        wandb.log({
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "kl_loss": train_kl / len(train_loader),
            "physics_loss": train_phys / len(train_loader),
            "alpha": criterion.alpha.item(),
            "beta": criterion.beta.item()
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = "models/bayesian_pinn.pth"
            if masked:
                os.makedirs("models/masked", exist_ok=True)
                save_path = "models/masked/bayesian_pinn.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl", type=float, default=0.1)
    parser.add_argument("--masked", action="store_true")
    args = parser.parse_args()
    
    train_bayesian_pinn(
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        learning_rate=args.lr, 
        kl_weight=args.kl,
        masked=args.masked
    )

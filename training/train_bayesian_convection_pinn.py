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

def train_bayesian_pinn(epochs=50, batch_size=32, learning_rate=1e-4, kl_weight=0.1):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="bayesian-pinn")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # BayesianResNet is frame-based (or last frame of sequence)
    # But Bioheat Loss works best with sequences if we want dT/dt.
    # However, BayesianResNet output is scalar.
    # So we will use the "Lumped Parameter" mode of Bioheat Loss (Newton's Law)
    # OR we can use Spatial Smoothing if we had a SpatialBayesianResNet.
    # Let's stick to Scalar Output + Temporal Sequence for Newton's Law.
    sequence_length = 5 
    
    print(f"Training Bayesian PINN on {device}")
    
    # Data
    print("Loading dataset...")
    # Use SequenceHeatmapDataset to match 4 output scalars
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
        raw_dir="data/level0_raw",
        sequence_length=sequence_length, 
        target_size=(64, 64),
        use_optical_flow=True 
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
            if len(batch) == 4:
                images, _, _, labels = batch
            else:
                 raise ValueError("Unexpected batch len")

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
            
            # Correct shape: (B, T, 4)
            predictions = preds_flat.view(B, T, 4)
            
            # Extract flow for convection
            raw_flow = images[:, :, 3:5, :, :] # (B, T, 2, H, W)
            # Downsample flow to match prediction size (4x4) if needed, but BayesianResNet outputs scalar (1x1)
            # Wait, BayesianResNet outputs scalar. So we can't do spatial convection on the output map.
            # But AdvancedBioHeatLoss handles scalar predictions by assuming lumped parameter model (no spatial gradients).
            # Convection requires spatial gradients.
            # So we can ONLY do Convection if the model outputs a MAP.
            # BayesianResNet outputs (B, 1).
            # So we cannot do Convection with the current BayesianResNet.
            # We need a BayesianSpatialResNet that outputs a map.
            
            # Let's check if we can modify BayesianResNet to output a map or use BayesianCNNLSTM which outputs scalar.
            # If we want Convection, we need Spatial Map.
            
            # For now, let's stick to the plan. If BayesianResNet outputs scalar, we can't use convection term in loss.
            # The loss function handles this:
            # if predictions.dim() == 4: ... convection_term ...
            # else: ... convection_term = 0.0
            
            # So adding flow here won't help unless we change the model to output a map.
            # Let's skip Bayesian Convection PINN for now as it requires a new model architecture (BayesianSpatialResNet).
            
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
                else: 
                     raise ValueError("Batch len")

                images = images.to(device)
                labels = labels.to(device).float()
                
                B, T, C, H, W = images.shape
                images_flat = images.view(B*T, C, H, W)
                preds_flat, _ = model(images_flat)
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
            torch.save(model.state_dict(), "models/bayesian_pinn.pth")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_bayesian_pinn(epochs=args.epochs)

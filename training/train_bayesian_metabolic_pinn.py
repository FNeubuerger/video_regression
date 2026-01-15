"""
Training script for Bayesian Metabolic Physics-Informed Neural Network.
Combines Bayesian Neural Networks (Uncertainty) with Metabolic Bioheat Loss.
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
from models.bayesian import BayesianCNNLSTM
from physics.bioheat_loss import AdvancedBioHeatLoss
from tqdm import tqdm
import wandb
import torchbnn as bnn

def train_bayesian_metabolic_pinn(epochs=50, batch_size=32, learning_rate=1e-4, kl_weight=0.1):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="bayesian-metabolic-pinn")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5 
    
    print(f"Training Bayesian Metabolic PINN on {device}")
    
    # Data
    print("Loading dataset...")
    # Use SequenceHeatmapDataset for 4 scalar targets
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
    print("Initializing BayesianCNNLSTM...")
    # Input: 5 channels (3 RGB + 2 Flow)
    model = BayesianCNNLSTM(frame_shape=(64, 64, 5)).to(device)
    
    # Advanced Bioheat Loss with Metabolic Rate
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01, 
        initial_conductivity=0.001,
        initial_metabolic_rate=0.005, # Enable metabolic term
        arterial_temp=37.0,
        learnable_params=True,
        dt=1.0
    ).to(device)
    
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
                 raise ValueError("Batch len")
            
            images = images.to(device) # (B, T, 5, 64, 64)
            labels = labels.to(device).float() # (B, T, 4)
            
            optimizer.zero_grad()
            
            # Bayesian Forward Pass
            predictions, kl_div = model(images) # (B, T, 4)
            # predictions = predictions.mean(dim=-1) # (B, T) -> REMOVED
            
            # 1. Physics Loss (includes MSE + Physics)
            phys_loss = criterion(predictions, labels)
            
            # 2. KL Divergence Loss
            kl = kl_div
            
            # Total Loss
            total_loss = phys_loss + (kl_weight * kl)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
            train_kl += kl.item()
            train_phys += phys_loss.item()
            
            progress.set_postfix(loss=total_loss.item(), kl=kl.item(), phys=phys_loss.item(), qm=f"{criterion.qm.item():.4f}")
            
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
                
                predictions, _ = model(images)
                
                phys_loss = criterion(predictions, labels)
                val_loss += phys_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        print(f"Physics Params: Alpha={criterion.alpha.item():.5f}, Beta={criterion.beta.item():.5f}, Qm={criterion.qm.item():.5f}")
        
        wandb.log({
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "kl_loss": train_kl / len(train_loader),
            "physics_loss": train_phys / len(train_loader),
            "alpha": criterion.alpha.item(),
            "beta": criterion.beta.item(),
            "qm": criterion.qm.item()
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/bayesian_metabolic_pinn.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl", type=float, default=0.1)
    args = parser.parse_args()
    
    train_bayesian_metabolic_pinn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        kl_weight=args.kl
    )

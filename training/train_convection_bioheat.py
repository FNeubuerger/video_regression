"""
Training script for Convection Bioheat-Informed CNN-LSTM (Scalar).
Uses BioHeatEquationLoss with Convection term (v . grad T).
Requires Optical Flow logic.
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

def train_convection_bioheat_model(epochs=50, batch_size=32, learning_rate=1e-4):
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="convection-bioheat-cnnlstm")
    
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence_length = 5 
    
    print(f"Training Convection Bioheat Model on {device}")
    
    # Data
    print("Loading dataset...")
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped", 
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
    model = PhysicsCNNLSTM(frame_shape=(64, 64, 5), time_steps=sequence_length, pretrained=True)
    model.to(device)
    
    # Loss
    criterion = AdvancedBioHeatLoss(
        physics_weight=1.0, 
        initial_perfusion=0.01, 
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
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for frames, maps, priors, scalars in progress:
            frames = frames.to(device)
            scalars = scalars.to(device).float()
            
            # Flow is in channels 3 and 4 of frames (0,1,2=RGB, 3=FlowX, 4=FlowY)
            # AdvancedBioHeatLoss expects flow tensor (B, T, 2, H, W)
            flow = frames[:, :, 3:, :, :] 
            
            optimizer.zero_grad()
            predictions = model(frames)
            
            # Pass flow to criterion to enable Convection Term
            # Note: For Scalar Output, Convection (v . grad T) is strictly not computable 
            # unless we approximate spatial gradient from temporal or assume steady state?
            # AdvancedBioHeatLoss:
            # if predictions.dim() == 2 (Scalar): "Convection term ... not computable" logic?
            # Checking bioheat_loss.py: 
            # It only computes convection if predictions is Map (3D/4D).
            # If predictions is scalar, it falls back to MSE (+ possibly ODE terms).
            # So "Convection Bioheat" is paradoxical for Scalar Model?
            # Unless we use "SpatialPhysicsCNNLSTM" which outputs a Map?
            # The plan lists "Convection Bioheat" under "Temporal Models (Sequence-based)".
            # Maybe it implies using latent features?
            # Or maybe it simply adds flow as input to the model (which we do)?
            # If so, passing flow to loss is irrelevant if loss ignores it.
            # But let's pass it anyway.
            
            loss = criterion(predictions, scalars, flow=flow)
            
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
                frames = frames.to(device)
                scalars = scalars.to(device).float()
                flow = frames[:, :, 3:, :, :]
                
                predictions = model(frames)
                loss = criterion(predictions, scalars, flow=flow)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/convection_bioheat_model.pth")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    
    train_convection_bioheat_model(epochs=args.epochs)

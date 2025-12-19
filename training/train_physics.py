"""
Training script for Physics-Informed CNN-LSTM.
Uses PhysicsInformedLoss to enforce temporal smoothness.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
from physics import PhysicsCNNLSTM, PhysicsInformedLoss
from tqdm import tqdm

def train_physics_model():
    # Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 20
    sequence_length = 5 # Longer sequence to better use physics loss
    
    # Data
    print("Loading dataset...")
    dataset = TemperatureSequenceDataset(
        data_dir="data", 
        sequence_length=sequence_length, 
        image_size=(64, 64)
    )
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model
    print("Initializing PhysicsCNNLSTM...")
    model = PhysicsCNNLSTM(frame_shape=(64, 64, 3), time_steps=sequence_length, pretrained=True)
    model.to(device)
    
    # Physics Loss
    criterion = PhysicsInformedLoss(smoothness_weight=1.0) # High weight for smoothness
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training Loop
    print(f"Starting training on {device}...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device).float() # (batch,) - scalar label for sequence
            
            optimizer.zero_grad()
            
            # Forward
            # Model outputs (batch, time_steps)
            predictions = model(images)
            
            # Loss
            # We only have label for the LAST frame (or the sequence label)
            # PhysicsLoss handles this: MSE on last frame + Smoothness on sequence
            loss = criterion(predictions, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                predictions = model(images)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/physics_cnnlstm_model.pth")
            print("Saved best model.")

if __name__ == "__main__":
    train_physics_model()

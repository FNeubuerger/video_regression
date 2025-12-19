"""
Training script to fix PretrainedCNNLSTM performance.
Uses lower learning rate and differential learning rates.
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from utils.dataset import TemperatureSequenceDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
import argparse
import time
import json
from torch.amp import GradScaler, autocast

def create_model(model_type, frame_shape, time_steps):
    """Create and return the specified model."""
    if model_type == "cnnlstm":
        return CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "pretrained_cnnlstm":
        pretrained_cnn = resnet18(weights='IMAGENET1K_V1')
        # We don't need to modify fc here as PretrainedCNNLSTM strips it, 
        # but it doesn't hurt.
        return PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "simple_resnet":
        return SimpleResNet(frame_shape=frame_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_model_with_validation(model_instance, model_name, criterion_instance, optimizer_instance, 
                               train_loader, val_loader, device, num_epochs=50, patience=10, 
                               model_save_path=None):
    if model_save_path is None:
        model_save_path = f"models/{model_name}_fixed_model.pth"
    
    model_instance.to(device)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    history = {'train_loss': [], 'val_loss': [], 'epochs': 0, 'training_time': 0}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\n=== Training {model_name} (Fixed) ===")
    print(f"Device: {device}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model_instance.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
        
        for images, labels in train_progress:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer_instance.zero_grad()
            
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model_instance(images)
                    loss = criterion_instance(outputs, labels.float())
                scaler.scale(loss).backward()
                scaler.step(optimizer_instance)
                scaler.update()
            else:
                outputs = model_instance(images)
                loss = criterion_instance(outputs, labels.float())
                loss.backward()
                optimizer_instance.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model_instance.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if scaler is not None:
                    with autocast('cuda'):
                        outputs = model_instance(images)
                        loss = criterion_instance(outputs, labels.float())
                else:
                    outputs = model_instance(images)
                    loss = criterion_instance(outputs, labels.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model_instance.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered")
                break
                
    history['epochs'] = epoch + 1
    history['training_time'] = time.time() - start_time
    return history

def main():
    # Setup
    data_dir = "data"
    batch_size = 32 # Smaller batch size for stability
    image_size = (64, 64)
    sequence_length = 3
    
    dataset = TemperatureSequenceDataset(
        data_dir, 
        sequence_length=sequence_length, 
        image_size=image_size
    )
    
    # Split
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model: PretrainedCNNLSTM
    print("Initializing PretrainedCNNLSTM...")
    model = create_model("pretrained_cnnlstm", (image_size[0], image_size[1], 3), sequence_length)
    model.to(device)
    
    # Differential Learning Rates
    # Backbone (CNN): Low LR to preserve features
    # Head (LSTM + FC): Higher LR to learn the task
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if "cnn" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},  # Very low LR for ResNet
        {'params': head_params, 'lr': 1e-3}       # Standard LR for LSTM/FC
    ], weight_decay=1e-4)
    
    criterion = nn.MSELoss()
    
    # Train
    train_model_with_validation(
        model, "pretrained_cnnlstm", criterion, optimizer, 
        train_loader, val_loader, device, num_epochs=20, patience=5
    )

if __name__ == "__main__":
    main()

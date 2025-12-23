"""
Comprehensive training script for all temperature regression models.

This script trains and compares:
1. CNNLSTM - Custom CNN-LSTM model
2. PretrainedCNNLSTM - ResNet18 + LSTM model  
3. SimpleResNet - Simple ResNet18 for single frame regression
4. PhysicsCNNLSTM - Physics-Informed CNN-LSTM model

All models are trained with early stopping and the results are saved for comparison.
"""

import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from physics import PhysicsCNNLSTM, PhysicsInformedLoss
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
import wandb

def create_model(model_type, frame_shape, time_steps):
    """Create and return the specified model."""
    if model_type == "cnnlstm":
        return CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "pretrained_cnnlstm":
        pretrained_cnn = resnet18(weights='IMAGENET1K_V1')
        pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)
        return PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "simple_resnet":
        return SimpleResNet(frame_shape=frame_shape)
    elif model_type == "physics_cnnlstm":
        return PhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_model_with_validation(model_instance, model_name, criterion_instance, optimizer_instance, 
                               train_loader, val_loader, device, num_epochs=50, patience=10, 
                               model_save_path=None):
    """
    Train a model with validation and early stopping.
    
    Returns:
        dict: Training history and final metrics
    """
    if model_save_path is None:
        model_save_path = f"models/{model_name}_model.pth"
    
    # Ensure model is on device
    model_instance.to(device)
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # GPU optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': 0,
        'best_epoch': 0,
        'training_time': 0
    }
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\n=== Training {model_name} ===")
    print(f"Device: {device}")
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model_instance.train()
        train_loss = 0.0
        train_progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training")
        
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer_instance.zero_grad()
            
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model_instance(images)
                    # Handle physics model output (sequence) vs scalar label
                    if model_name == "physics_cnnlstm":
                        loss = criterion_instance(outputs, labels.float())
                    else:
                        loss = criterion_instance(outputs, labels.float())
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer_instance)
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
                
                scaler.step(optimizer_instance)
                scaler.update()
            else:
                outputs = model_instance(images)
                if model_name == "physics_cnnlstm":
                    loss = criterion_instance(outputs, labels.float())
                else:
                    loss = criterion_instance(outputs, labels.float())
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
                
                optimizer_instance.step()
            
            train_loss += loss.item()
            train_progress.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model_instance.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation")
            for images, labels in val_progress:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                if scaler is not None:
                    with autocast('cuda'):
                        outputs = model_instance(images)
                        if model_name == "physics_cnnlstm":
                            loss = criterion_instance(outputs, labels.float())
                        else:
                            loss = criterion_instance(outputs, labels.float())
                else:
                    outputs = model_instance(images)
                    if model_name == "physics_cnnlstm":
                        loss = criterion_instance(outputs, labels.float())
                    else:
                        loss = criterion_instance(outputs, labels.float())
                
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to WandB
        wandb.log({
            f"{model_name}/train_loss": avg_train_loss,
            f"{model_name}/val_loss": avg_val_loss,
            "epoch": epoch + 1
        })
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epochs'] = epoch + 1
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            history['best_epoch'] = epoch + 1
            
            # Save best model
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model_instance.state_dict(), model_save_path)
            print(f"✓ New best model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    training_time = time.time() - start_time
    history['training_time'] = training_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {history['best_epoch']}")
    
    return history


def main():
    # Set working directory to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    print(f"Working directory set to: {project_root}")

    # Load config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found, using defaults.")
        config = {
            "force_rerun": {},
            "training_params": {"epochs": 50, "batch_size": 128, "patience": 10}
        }

    parser = argparse.ArgumentParser(description="Train all temperature regression models")
    parser.add_argument("--epochs", type=int, default=config["training_params"].get("epochs", 50), help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=config["training_params"].get("patience", 10), help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=config["training_params"].get("batch_size", 128), help="Batch size")
    parser.add_argument("--models", nargs='+', 
                       choices=['cnnlstm', 'pretrained_cnnlstm', 'simple_resnet', 'physics_cnnlstm', 'all'],
                       default=['all'], help="Models to train")
    args = parser.parse_args()
    
    # Model parameters - optimized for speed
    # 3 RGB channels + 2 Optical Flow channels = 5 channels
    frame_shape = (64, 64, 5)
    time_steps = 3
    
    print("=== Preparing Dataset ===")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = TemperatureSequenceDataset(
        data_dir="data",
        sequence_length=time_steps,
        transform=transform,
        image_size=(64, 64),
        use_optical_flow=True
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.15 * total_size)   # 15% for validation
    test_size = total_size - train_size - val_size  # 15% for testing
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine which models to train
    if 'all' in args.models:
        models_to_train = ['cnnlstm', 'pretrained_cnnlstm', 'simple_resnet', 'physics_cnnlstm']
    else:
        models_to_train = args.models
    
    # Initialize WandB
    wandb.init(project="video-temperature-regression", name="standard-models-benchmark")
    
    # Training results
    results = {}
    
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")

        # Check if we should skip this model
        model_path = f"checkpoints/{model_type}_model.pth"
        should_rerun = config["force_rerun"].get(model_type, False)
        
        # Force rerun if requested via args (implicit in this script usage)
        # But we respect the config file primarily.
        # If the user moved files to archive, os.path.exists will be false, so it will retrain.
        
        if os.path.exists(model_path) and not should_rerun:
            print(f"Model {model_type} already exists and force_rerun is False. Skipping.")
            continue
        
        # Create model
        model = create_model(model_type, frame_shape, time_steps)
        
        # Create criterion and optimizer
        if model_type == "physics_cnnlstm":
            # Use PhysicsInformedLoss for the physics model
            criterion = PhysicsInformedLoss(smoothness_weight=0.1)
        else:
            criterion = nn.MSELoss()
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        
        # Train model
        try:
            history = train_model_with_validation(
                model_instance=model,
                model_name=model_type,
                criterion_instance=criterion,
                optimizer_instance=optimizer,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=args.epochs,
                patience=args.patience
            )
            
            results[model_type] = history
            
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            results[model_type] = {"error": str(e)}
    
    wandb.finish()
    
    # Save results
    
    # Save results
    results_file = "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    
    for model_type, result in results.items():
        if "error" in result:
            print(f"{model_type:20s}: FAILED - {result['error']}")
        else:
            print(f"{model_type:20s}: {result['epochs']:3d} epochs, "
                  f"Best Val Loss: {min(result['val_loss']):.4f}, "
                  f"Time: {result['training_time']:.1f}s")
    
    print(f"\nResults saved to {results_file}")
    print("Run 'python comprehensive_evaluation.py' to evaluate all models!")


if __name__ == "__main__":
    main()
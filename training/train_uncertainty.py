import sys
import os
import argparse
import torch
import torch.nn as nn
import torchbnn as bnn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import SimpleResNet
from models.bayesian import BayesianResNet, FullBayesianResNet
from utils.sequence_dataset import SequenceHeatmapDataset
from torchvision import transforms
import wandb

def train_ensemble(num_models=5, epochs=20, batch_size=128):
    wandb.init(project="video-temperature-regression", name="ensemble-training")
    print(f"\n{'='*50}")
    print(f"Training Ensemble of {num_models} SimpleResNet models")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        sequence_length=5,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=True # Ensemble usually benefits from masking
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    os.makedirs("checkpoints/ensemble", exist_ok=True)
    
    # Frame shape for 5 channels (3 RGB + 2 Flow)
    frame_shape = (64, 64, 5)

    for i in range(num_models):
        print(f"\nTraining Ensemble Member {i+1}/{num_models}")
        model = SimpleResNet(frame_shape=frame_shape).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            
            for batch in progress:
                if len(batch) == 4:
                    images, labels_raw, mask, scalars = batch
                elif len(batch) == 3:
                     images, labels_raw, mask = batch
                     scalars = None
                else: 
                     images, labels_raw = batch
                     mask = None
                     scalars = None

                images = images.to(device)
                
                # Use scalars as label if available, else derive from heatmap
                if scalars is not None:
                    labels = scalars.to(device)
                    if labels.dim() == 3:
                        labels = labels[:, -1, :]
                else:
                    labels = labels_raw.to(device)
                    if labels.dim() == 4:
                        labels = labels.amax(dim=(1,2,3))
                    elif labels.dim() == 5:
                        labels = labels.amax(dim=(2,3,4))

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                progress.set_postfix(loss=loss.item())
            
            wandb.log({f"ensemble_model_{i}/loss": train_loss/len(train_loader), "epoch": epoch+1})
        
        # Save model
        torch.save(model.state_dict(), f"checkpoints/ensemble/model_{i}.pth")
        print(f"Saved checkpoints/ensemble/model_{i}.pth")
    wandb.finish()

def train_bayesian(epochs=30, batch_size=128, kl_weight=0.1):
    wandb.init(project="video-temperature-regression", name="bayesian-head-training")
    print(f"\n{'='*50}")
    print(f"Training Bayesian ResNet")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        sequence_length=5,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=True 
    )
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Frame shape for 5 channels (3 RGB + 2 Flow)
    frame_shape = (64, 64, 5)
    model = BayesianResNet(frame_shape=frame_shape).to(device)
    
    mse_loss = nn.MSELoss()
    # KL Loss defined in model usually, but also can use torchbnn
    # BayesianResNet now returns (pred, kl) tuple
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress:
            if len(batch) == 4:
                images, labels_raw, mask, scalars = batch
            else:
                 # Should not happen with new dataset class but fallback
                 images, labels_raw = batch[0], batch[1]
                 scalars = None

            images = images.to(device)
            
            if scalars is not None:
                labels = scalars.to(device)
                if labels.dim() == 3:
                    labels = labels[:, -1, :]
            else:
                labels = labels_raw.to(device)
                if labels.dim() == 4:
                     labels = labels.amax(dim=(1,2,3))

            optimizer.zero_grad()
            outputs, kl = model(images)
            
            mse = mse_loss(outputs, labels.float())
            loss = mse + kl_weight * kl
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item(), mse=mse.item(), kl=kl.item())
            
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        wandb.log({"bayesian_head/loss": avg_loss, "epoch": epoch+1})
        
    torch.save(model.state_dict(), "checkpoints/bayesian_resnet.pth")
    print("Saved checkpoints/bayesian_resnet.pth")
    wandb.finish()

def train_full_bayesian(epochs=30, batch_size=128, kl_weight=0.1):
    wandb.init(project="video-temperature-regression", name="full-bayesian-training")
    print(f"\n{'='*50}")
    print(f"Training FULL Bayesian ResNet")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        sequence_length=5,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=True
    )
    
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = FullBayesianResNet(frame_shape=(64, 64, 5)).to(device)
    mse_loss = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        
        for batch in progress:
            if len(batch) == 4:
                images, labels_raw, mask, scalars = batch
            else:
                 images, labels_raw = batch[0], batch[1]
                 scalars = None

            images = images.to(device)
            if scalars is not None:
                labels = scalars.to(device)
                if labels.dim() == 3:
                     labels = labels[:, -1, :]
            else:
                labels = labels_raw.to(device)
                if labels.dim() == 4:
                     labels = labels.amax(dim=(1,2,3))
            
            optimizer.zero_grad()
            outputs, kl = model(images)
            
            mse = mse_loss(outputs, labels.float())
            loss = mse + kl_weight * kl
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item(), mse=mse.item(), kl=kl.item())
            
        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
        wandb.log({"full_bayesian/loss": avg_loss, "epoch": epoch+1})
        
    torch.save(model.state_dict(), "checkpoints/full_bayesian_resnet.pth")
    print("Saved checkpoints/full_bayesian_resnet.pth")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ensemble", "bayesian", "full_bayesian", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    
    if args.mode in ["ensemble", "all"]:
        train_ensemble(epochs=args.epochs)
    if args.mode in ["bayesian", "all"]:
        train_bayesian(epochs=args.epochs)
    if args.mode in ["full_bayesian", "all"]:
        train_full_bayesian(epochs=args.epochs)

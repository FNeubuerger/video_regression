import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.latent_ltc import LatentLTC_UNet
from models.conv_ltc import ConvLTC_Model
from utils.sequence_dataset import SequenceHeatmapDataset
from physics.hybrid_loss import BioheatHybridLoss

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4) # Higher LR for RNNs sometimes needed
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--ncp_units', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--output_dir', default='checkpoints/ltc_unet')
    parser.add_argument('--run_name', default='ltc_unet_seq16')
    parser.add_argument('--limit_samples', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_physics_prior', action='store_true', help="Disable Physics Prior")
    parser.add_argument('--lambda_physics', type=float, default=1e-4)
    parser.add_argument('--model_type', type=str, default='latent_ltc', choices=['latent_ltc', 'conv_ltc'], help="Choose architecture: latent_ltc or conv_ltc")
    parser.add_argument('--variational', action='store_true', help="Use Bayesian/Variational Encoder")
    parser.add_argument('--beta_kl', type=float, default=0.01, help="Weight for KL Divergence Loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Sequence Dataset
    full_dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        sequence_length=args.sequence_length,
        stride=args.stride,
        use_physics_prior=not args.no_physics_prior
    )

    if args.limit_samples:
        indices = torch.randperm(len(full_dataset))[:args.limit_samples]
        full_dataset = torch.utils.data.Subset(full_dataset, indices)
    
    # Split
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, # Validation can use same batch size
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 2. Model
    if args.model_type == 'conv_ltc':
        print("Initializing ConvLTC Model (Spatially Continuous)...")
        model = ConvLTC_Model(
            input_channels=3,
            hidden_channels=32 # Using fixed hidden size for now, could be arg
        ).to(device)
    else:
        print("Initializing Latent LTC Model...")
        model = LatentLTC_UNet(
            n_channels=3, 
            latent_dim=args.latent_dim, 
            ncp_units=args.ncp_units,
            variational=args.variational
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Loss
    hybrid_criterion = BioheatHybridLoss(
        lambda_physics=args.lambda_physics, 
        dx=0.0006, 
        device=device
    )
    
    # 4. WandB
    wandb.init(project="video-regression-part3", name=args.run_name, config=args)
    
    best_val_loss = float('inf')
    
    print(f"Starting training: {len(train_dataset)} train seqs, {len(val_dataset)} val seqs")
    
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss_accum = 0.0
        train_mse_accum = 0.0
        train_phys_accum = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for frames, targets, priors in pbar:
            # frames: (B, T, 3, H, W)
            # targets: (B, T, 1, H, W)
            # priors: (B, T, 1, H, W)
            
            frames = frames.to(device)
            targets = targets.to(device)
            priors = priors.to(device)
            
            optimizer.zero_grad()
            
            # Forward (LTC handles time internally)
            # Output: (B, T, 1, H, W) - Represents DELTA if prior used
            if args.model_type == 'latent_ltc' and args.variational:
                outputs, kl_loss = model(frames)
            else:
                outputs = model(frames)
                kl_loss = torch.tensor(0.0).to(device)
            
            if not args.no_physics_prior:
                final_preds = outputs + priors
            else:
                final_preds = outputs
                
            # Flatten for Loss
            # (B, T, 1, H, W) -> (B*T, 1, H, W)
            preds_flat = final_preds.view(-1, 1, 64, 64)
            targets_flat = targets.view(-1, 1, 64, 64)
            
            mask_flat = (targets_flat > 0).float()
            
            # Compute Loss
            loss, mse_val, phys_val = hybrid_criterion(preds_flat, targets_flat, mask_flat)
            
            total_loss = loss + args.beta_kl * kl_loss

            total_loss.backward()
            optimizer.step()
            
            train_loss_accum += total_loss.item()
            train_mse_accum += mse_val
            train_phys_accum += phys_val
            
            pbar.set_postfix({'loss': total_loss.item(), 'mse': mse_val, 'kl': kl_loss.item()})
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        
        with torch.no_grad():
            for frames, targets, priors in val_loader:
                frames = frames.to(device)
                targets = targets.to(device)
                priors = priors.to(device)
                
                if args.model_type == 'latent_ltc' and args.variational:
                    outputs, kl_loss = model(frames)
                else:
                    outputs = model(frames)
                    kl_loss = torch.tensor(0.0).to(device)
                
                if not args.no_physics_prior:
                    final_preds = outputs + priors
                else:
                    final_preds = outputs
                    
                preds_flat = final_preds.view(-1, 1, 64, 64)
                targets_flat = targets.view(-1, 1, 64, 64)
                mask_flat = (targets_flat > 0).float()
                
                loss, mse_val, _ = hybrid_criterion(preds_flat, targets_flat, mask_flat)
                total_loss = loss + args.beta_kl * kl_loss
                
                val_loss_accum += total_loss.item()
                val_mse_accum += mse_val
                
        avg_val_loss = val_loss_accum / len(val_loader)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_mse': train_mse_accum / len(train_loader),
            'train_physics_loss': train_phys_accum / len(train_loader),
            'val_loss': avg_val_loss,
            'val_mse': val_mse_accum / len(val_loader)
        })
        
        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, avg_val_loss, 
                          os.path.join(args.output_dir, 'best_model.pth'))
                          
        save_checkpoint(model, optimizer, epoch, avg_val_loss, 
                      os.path.join(args.output_dir, 'latest_model.pth'))

if __name__ == "__main__":
    main()

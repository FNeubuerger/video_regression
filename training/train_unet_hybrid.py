import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
import sys

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dense_heads import ResNetUNet
from utils.heatmap_dataset import TemperatureHeatmapDataset
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', default='checkpoints/unet_hybrid')
    parser.add_argument('--run_name', default='unet_hybrid_physics')
    parser.add_argument('--limit_samples', type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_physics_prior', action='store_true', help="Disable Physics Prior (Gaussian base)")
    parser.add_argument('--lambda_physics', type=float, default=1e-4, help="Weight for Physics PDE Loss")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    full_dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        use_physics_prior=not args.no_physics_prior
    )

    if args.limit_samples:
        print(f"WARN: Limiting dataset to {args.limit_samples} samples for debugging.")
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
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 2. Model
    model = ResNetUNet(n_channels=3, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. Loss
    # We use the new BioheatHybridLoss
    hybrid_criterion = BioheatHybridLoss(
        lambda_physics=args.lambda_physics, 
        dx=0.0006,  # 0.6mm/px (approx for 64x64 crop of ultrasonic)
        device=device
    )
    
    # 4. WandB
    wandb.init(project="video-regression-part2", name=args.run_name, config=args)
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs with Lambda Physics = {args.lambda_physics}...")
    
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss_accum = 0.0
        mse_loss_accum = 0.0
        phys_loss_accum = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for frames, targets, masks, _, priors in train_pbar:
            frames = frames.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            priors = priors.to(device)
            
            optimizer.zero_grad()
            
            # RESIDUAL LEARNING
            delta = model(frames)
            preds = delta + priors
            
            # Hybrid Loss
            # returns (total_loss, mse_item, physics_item)
            loss, mse_item, phys_item = hybrid_criterion(preds, targets, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            mse_loss_accum += mse_item
            phys_loss_accum += phys_item
            
            train_pbar.set_postfix({'loss': loss.item(), 'mse': mse_item, 'phys': phys_item})
            
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_mse = mse_loss_accum / len(train_loader)
        avg_train_phys = phys_loss_accum / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss_accum = 0.0
        val_mse_accum = 0.0
        val_phys_accum = 0.0
        
        # Visualize one batch
        vis_batch = None
        
        with torch.no_grad():
            for i, (frames, targets, masks, _, priors) in enumerate(tqdm(val_loader, desc="[Val]")):
                frames = frames.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                priors = priors.to(device)
                
                delta = model(frames)
                preds = delta + priors
                
                loss, mse_item, phys_item = hybrid_criterion(preds, targets, masks)
                
                val_loss_accum += loss.item()
                val_mse_accum += mse_item
                val_phys_accum += phys_item
                
                if i == 0:
                    vis_batch = (frames, targets, preds, priors)

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_mse = val_mse_accum / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} (MSE: {avg_train_mse:.4f} Phys: {avg_train_phys:.4f}) | Val Loss {avg_val_loss:.4f}")
        
        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_mse_loss": avg_train_mse,
            "train_physics_loss": avg_train_phys,
            "val_loss": avg_val_loss,
            "val_mse_loss": avg_val_mse # Most comparable to other baselines
        })
        
        # Visualization Log
        if vis_batch:
            frames, targets, preds, priors = vis_batch
            img = frames[0].cpu()
            # Unnormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = img.clamp(0, 1)
            
            pred_map = preds[0, 0].cpu().numpy()
            target_map = targets[0, 0].cpu().numpy()
            prior_map = priors[0, 0].cpu().numpy()

            wandb.log({
                "example_frame": wandb.Image(img),
                "example_pred": wandb.Image(pred_map, caption=f"Pred Map (Val Loss: {avg_val_loss:.2f})"),
                "example_prior": wandb.Image(prior_map, caption="Physics Prior"),
                "example_target": wandb.Image(target_map, caption="Sparse Target")
            })

        # Checkpoints
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.output_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, avg_val_loss, save_path)
            
        # Save Latest
        save_path = os.path.join(args.output_dir, "latest_model.pth")
        save_checkpoint(model, optimizer, epoch, avg_val_loss, save_path)
        
    wandb.finish()

if __name__ == "__main__":
    main()

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

def sparse_mse_loss(pred, target, mask):
    """
    Computes MSE only at masked locations.
    pred: (B, 1, H, W)
    target: (B, 1, H, W) -- sparse values
    mask: (B, 1, H, W) -- 1 at valid pixels, 0 elsewhere
    """
    diff = (pred - target) * mask
    loss = (diff ** 2).sum() / mask.sum().clamp(min=1.0)
    return loss

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
    parser.add_argument('--output_dir', default='checkpoints/unet_sparse')
    parser.add_argument('--run_name', default='resnet_unet_sparse_baseline')
    parser.add_argument('--limit_samples', type=int, default=None, help="Limit dataset size for debugging")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_physics_prior', action='store_true', help="Disable Physics Prior (Gaussian)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    full_dataset = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),  # Downscale for efficiency during development
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
    
    # 3. WandB
    wandb.init(project="video-regression-part2", name=args.run_name, config=args)
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # TRAIN
        model.train()
        train_loss_accum = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for frames, targets, masks, _, priors in train_pbar:
            frames = frames.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            priors = priors.to(device)
            
            optimizer.zero_grad()
            
            # RESIDUAL LEARNING: Pred = Base + Prior
            # The model predicts the DELTA.
            delta = model(frames)
            preds = delta + priors
            
            loss = sparse_mse_loss(preds, targets, masks)
            loss.backward()
            optimizer.step()
            
            train_loss_accum += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # VALIDATION
        model.eval()
        val_loss_accum = 0.0
        
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
                
                loss = sparse_mse_loss(preds, targets, masks)
                val_loss_accum += loss.item()
                
                if i == 0:
                    vis_batch = (frames, targets, preds, priors)

        avg_val_loss = val_loss_accum / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f}")
        
        # Logging
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        # Visualization Log
        if vis_batch:
            frames, targets, preds, priors = vis_batch
            # Log first image in batch
            img = frames[0].cpu()
            # Unnormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = img.clamp(0, 1)
            
            pred_map = preds[0, 0].cpu().numpy()
            target_map = targets[0, 0].cpu().numpy() # Sparse, mostly zeros
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
            print("Saved Best Model.")
            
        # Save Latest
        save_path = os.path.join(args.output_dir, "latest_model.pth")
        save_checkpoint(model, optimizer, epoch, avg_val_loss, save_path)
        
    wandb.finish()

if __name__ == "__main__":
    main()

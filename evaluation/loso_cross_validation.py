"""
Leave-One-Sequence-Out (LOSO) Cross-Validation Script.

This script performs rigorous validation by training on (N-1) sequences 
and testing on the held-out sequence. This proves the spatial and 
setup generalization of our temperature regression models.
"""

import os
import torch
import numpy as np
import pandas as pd
import argparse
import sys
import json
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
from models.backbones import CNNLSTM, SimpleResNet, PretrainedCNNLSTM, SpatialResNet
from models.bayesian import BayesianResNet, FullBayesianResNet, BayesianSpatialResNet, BayesianCNNLSTM
from models.conv_ltc import ConvLTC
from physics.models import SpatialPhysicsCNNLSTM, PhysicsCNNLSTM
from physics.loss import PhysicsInformedLoss
from physics.bioheat_loss import AdvancedBioHeatLoss
from training.train_all_models import train_model_with_validation

def run_loso_fold(holdout_seq, model_type, args):
    """Run a single LOSO fold holding out holdout_seq."""
    print(f"\n{'='*60}")
    print(f"FOLD: Holding out {holdout_seq}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = TemperatureSequenceDataset(
        data_dir="data",
        sequence_length=5,
        transform=transform,
        use_optical_flow=True,
        image_size=(64, 64),
        use_artifact_masking=args.masked
    )
    
    # Split based on sequence names
    train_indices = []
    test_indices = []
    
    for i, (paths, _) in enumerate(full_dataset.sequences):
        if holdout_seq in paths[0]:
            test_indices.append(i)
        else:
            train_indices.append(i)
            
    if not test_indices:
        print(f"Warning: No samples found for sequence {holdout_seq}. Skipping fold.")
        return None

    # Further split train into train/val (85/15)
    np.random.shuffle(train_indices)
    split = int(0.85 * len(train_indices))
    val_indices = train_indices[split:]
    train_indices = train_indices[:split]
    
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)
    
    print(f"Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    frame_shape = (64, 64, 5)
    time_steps = 5
    
    if model_type == "ConvectionBioheat":
        # The ultimate spatial physics model (Issue #41, #42)
        model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
        criterion = AdvancedBioHeatLoss(
            physics_weight=0.1, 
            spatial_params=True, 
            learnable_params=True,
            frame_shape=(4, 4)
        )
    elif model_type == "SpatialPhysicsCNNLSTM":
        model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
        criterion = AdvancedBioHeatLoss(physics_weight=0.1, spatial_params=True, learnable_params=True)
    elif model_type == "PhysicsCNNLSTM":
        model = PhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
        criterion = PhysicsInformedLoss(physics_weight=0.1)
    elif model_type == "CNNLSTM":
        model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
        criterion = torch.nn.MSELoss()
    elif model_type == "PretrainedCNNLSTM":
        # We need a dummy backbone for the constructor in backbones.py
        from torchvision.models import resnet18
        backbone = resnet18()
        backbone.fc = torch.nn.Linear(512, 1)
        model = PretrainedCNNLSTM(backbone, frame_shape=frame_shape, time_steps=time_steps)
        criterion = torch.nn.MSELoss()
    elif model_type == "SimpleResNet":
        model = SimpleResNet(frame_shape=frame_shape)
        criterion = torch.nn.MSELoss()
    elif model_type == "SpatialResNet":
        model = SpatialResNet(frame_shape=frame_shape)
        criterion = torch.nn.MSELoss()
    elif model_type == "BayesianResNet":
        model = BayesianResNet(frame_shape=frame_shape)
        criterion = torch.nn.MSELoss()
    elif model_type == "FullBayesianResNet":
        model = FullBayesianResNet(frame_shape=frame_shape)
        criterion = torch.nn.MSELoss()
    elif model_type == "BayesianCNNLSTM":
        model = BayesianCNNLSTM(frame_shape=frame_shape)
        criterion = torch.nn.MSELoss()
    elif model_type == "ConvLTC":
        model = ConvLTC(in_channels=5, hidden_channels=32)
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Ensure criterion is on device
    criterion = criterion.to(device)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Path for fold checkpoint
    os.makedirs(f"checkpoints/loso/{model_type}", exist_ok=True)
    save_path = f"checkpoints/loso/{model_type}/fold_{holdout_seq}.pth"
    
    # Train
    history = train_model_with_validation(
        model_instance=model,
        model_name=f"{model_type}_fold_{holdout_seq}",
        criterion_instance=criterion,
        optimizer_instance=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        patience=5,
        model_save_path=save_path,
        masked=args.masked
    )
    
    # Evaluate on held-out sequence
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    is_bayesian = "Bayesian" in model_type
    mc_samples = 10 if is_bayesian else 1
    
    errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Fold"):
            if len(batch) == 3:
                imgs, gt, mask = batch
                if args.masked: imgs = imgs * (1.0 - mask.unsqueeze(1))
            else:
                imgs, gt = batch
            
            imgs = imgs.to(device)
            
            # Monte Carlo sampling for Bayesian models
            batch_samples = []
            for _ in range(mc_samples):
                out = model(imgs)
                
                # Handle multiple outputs (Issue #41)
                if isinstance(out, tuple):
                    out = out[0]
                
                # Reduce if spatial (Spatial Map is (B, T, H, W) or (B, H, W))
                if out.dim() == 4: 
                    # Sequence of maps: Take last time step and average spatial
                    out = out[:, -1].mean(dim=(1, 2))
                elif out.dim() == 3:
                    # Single map: average spatial
                    out = out.mean(dim=(1, 2))
                elif out.dim() == 2:
                    # Basic Sequence (B, T) or Bayesian Batch (B, 1)
                    if out.shape[1] > 1:
                        out = out[:, -1]
                    else:
                        out = out.squeeze(-1)
                
                batch_samples.append(out.cpu().numpy())
            
            # Mean across MC samples
            out_mean = np.mean(batch_samples, axis=0) # (B,)
            gt_flat = gt.numpy().flatten()
            
            errors.extend((out_mean - gt_flat).tolist())
            
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))
    
    print(f"Fold {holdout_seq} Results: MAE={mae:.4f} K, RMSE={rmse:.4f} K")
    return {"fold": holdout_seq, "mae": float(mae), "rmse": float(rmse)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ConvectionBioheat")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--masked", action="store_true")
    args = parser.parse_args()
    
    # Identify sequences (limiting to valid sequence folders in our phantom study)
    seq_ids = [f"sequence_{i}" for i in [1, 2, 3, 5, 6, 7, 8]]
    
    results = []
    for seq_id in seq_ids:
        try:
            res = run_loso_fold(seq_id, args.model, args)
            if res:
                results.append(res)
        except Exception as e:
            print(f"Error in fold {seq_id}: {e}")
            
    # Aggregate
    if not results:
        print("No results collected.")
        return
        
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("-" * 60)
    print(f"MEAN MAE:  {df['mae'].mean():.4f} \pm {df['mae'].std():.4f} K")
    print(f"MEAN RMSE: {df['rmse'].mean():.4f} \pm {df['rmse'].std():.4f} K")
    
    os.makedirs("results", exist_ok=True)
    output_path = f"results/loso_{args.model}_{'masked' if args.masked else 'unmasked'}.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()

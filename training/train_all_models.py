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
from physics.models import SpatialPhysicsCNNLSTM
from physics.bioheat_loss import AdvancedBioHeatLoss
from utils.sequence_dataset import SequenceHeatmapDataset
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
        pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 4)
        return PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "simple_resnet":
        return SimpleResNet(frame_shape=frame_shape)
    elif model_type == "physics_cnnlstm":
        return PhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
    elif model_type == "spatial_physics_cnnlstm":
        return SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _pool_target_for_bioheat(labels, criterion):
    """Reduce a dense heatmap target to the (H, W) the bioheat loss expects.

    ConvectionBioheat / SpatialPhysicsCNNLSTM emit a small ``(B, T, h, w)``
    coarse field (often ``4x4``).  Our dataset's ``labels_raw`` is the full
    ``(B, T, 1, H, W)`` heatmap (typically ``64x64``).  Adaptive-pool the
    heatmap down to the criterion's ``frame_shape`` so they line up for
    elementwise MSE inside ``AdvancedBioHeatLoss``.
    """
    import torch.nn.functional as F

    target_hw = getattr(criterion, "frame_shape", None)
    if target_hw is None:
        return labels  # nothing to do

    if labels.dim() == 5:
        # (B, T, C, H, W) -> pool spatial -> (B, T, C, h, w) -> drop C
        B, T, C, H, W = labels.shape
        if (H, W) == tuple(target_hw):
            return labels.squeeze(2) if C == 1 else labels.mean(dim=2)
        pooled = F.adaptive_avg_pool2d(labels.reshape(B * T * C, 1, H, W), target_hw)
        pooled = pooled.reshape(B, T, C, *target_hw)
        return pooled.squeeze(2) if C == 1 else pooled.mean(dim=2)
    if labels.dim() == 4:
        # (B, T, H, W) sequence without channel dim
        B, T, H, W = labels.shape
        if (H, W) == tuple(target_hw):
            return labels
        pooled = F.adaptive_avg_pool2d(labels.reshape(B * T, 1, H, W), target_hw)
        return pooled.reshape(B, T, *target_hw)
    return labels


def align_outputs_target(outputs, target):
    """Coerce ``(outputs, target)`` into broadcast-compatible shapes for MSE.

    Handles every output rank we see across the model zoo so that the
    plain-MSE branch never explodes with a shape mismatch:

    * ``(B, T, 4, 4)`` ConvectionBioheat spatial map -> mean spatial -> ``(B, T)``.
    * ``(B, 4, 4)``   single spatial map           -> mean spatial -> ``(B,)``.
    * ``(B, T, C, H, W)`` spatial sequence         -> last frame, mean spatial.
    * ``(B, T, H, W)``    spatial sequence (no C)  -> last frame, mean spatial.
    * ``(B, H, W)``       single spatial map       -> mean spatial -> ``(B,)``.
    * ``(B, T, K)`` sensor-sequence (small K)      -> last frame -> ``(B, K)``.
    * ``(B, T)``    scalar sequence                -> last frame -> ``(B,)``.
    * ``(B, 1)`` / ``(B,)`` scalar -> reshape target to match.
    * Target rank greater than outputs -> take last frame of target.

    Both tensors are returned with matching ranks.  The function never
    moves data across devices; callers should call ``.to(device)`` first.
    """

    # 1) Reduce 5D spatial sequences to per-sample scalar.
    if outputs.dim() == 5:
        # (B, T, C, H, W) -> last frame, then mean over (C, H, W)
        outputs = outputs[:, -1].mean(dim=(1, 2, 3))

    # 2) Reduce 4D outputs: distinguish (B, T, H, W) from (B, T, K) and from (B, 4, 4).
    if outputs.dim() == 4:
        # ConvectionBioheat returns (B, T, 4, 4).  Keep T axis intact so it can
        # match a (B, T)-shaped target, then mean over the 4x4 grid.
        if outputs.shape[-1] <= 8 and outputs.shape[-2] <= 8:
            outputs = outputs.mean(dim=(2, 3))  # (B, T)
        else:
            outputs = outputs[:, -1].mean(dim=(1, 2))  # (B,)

    # 3) Reduce 3D outputs: (B, T, K) sensor sequence vs (B, H, W) map vs (B, 4, 4).
    if outputs.dim() == 3:
        if outputs.shape[1:] == (4, 4):
            outputs = outputs.mean(dim=(1, 2))  # (B,)
        elif outputs.shape[-1] <= 8 and outputs.shape[-2] <= 8 and outputs.shape[-2] != outputs.shape[-1]:
            # (B, T, K) with K small  -> take last frame
            outputs = outputs[:, -1]
        elif outputs.shape[-1] > 16:
            # (B, H, W) spatial map -> mean
            outputs = outputs.mean(dim=(1, 2))
        else:
            # Ambiguous; default to last frame (works for K==H==W==4 too).
            outputs = outputs[:, -1]
            if outputs.dim() == 3:
                outputs = outputs.mean(dim=(1, 2))

    # 4) Now align target rank to outputs.
    # Heatmap target (B, T, 1, H, W) or (B, 1, H, W) -> reduce to sensor-max scalar.
    if target.dim() == 5:
        target = target.amax(dim=(2, 3, 4))  # (B, T)
    if target.dim() == 4:
        target = target.amax(dim=(1, 2, 3))  # (B,)

    # If outputs are scalar but target is (B, K), match by taking last sensor/time.
    if outputs.dim() == 1 and target.dim() >= 2:
        target = target.reshape(target.shape[0], -1)[:, -1]

    # If outputs are (B, K) and target is (B, T, K), take last time step.
    if outputs.dim() == 2 and target.dim() == 3:
        target = target[:, -1, :]

    # If outputs are (B, T) and target is (B, T, K), reduce target across sensors.
    if outputs.dim() == 2 and target.dim() == 3 and outputs.shape[1] == target.shape[1]:
        target = target.mean(dim=-1)

    # Final guard: if last dims still differ, squeeze trailing singletons.
    while outputs.dim() > 1 and outputs.shape[-1] == 1:
        outputs = outputs.squeeze(-1)
    while target.dim() > 1 and target.shape[-1] == 1:
        target = target.squeeze(-1)

    # Last resort: flatten both so MSELoss can at least compute.
    if outputs.shape != target.shape:
        outputs = outputs.reshape(-1)
        target = target.reshape(-1)
        n = min(outputs.numel(), target.numel())
        outputs = outputs[:n]
        target = target[:n]

    return outputs, target


def train_model_with_validation(model_instance, model_name, criterion_instance, optimizer_instance, 
                               train_loader, val_loader, device, num_epochs=50, patience=10, 
                               model_save_path=None, masked=False):
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
        
        for batch_idx, batch in enumerate(train_progress):
            scalars = None
            # Robust unpacking of batch
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 4:
                    images, labels_raw, mask, scalars = batch[0], batch[1], batch[2], batch[3]
                    # SequenceHeatmapDataset's third tensor is a 5D prior heatmap,
                    # not an artifact mask the bioheat loss can use.
                    if isinstance(mask, torch.Tensor) and mask.dim() != 4:
                        mask = None
                elif len(batch) == 3:
                    images, labels_raw, mask = batch[0], batch[1], batch[2]
                    if isinstance(mask, torch.Tensor) and mask.dim() != 4:
                        mask = None
                else:
                    images, labels = batch[0], batch[1]
                    mask = None
                    labels_raw = labels # Align variable name
            else:
                # Raise error or handle unexpected type
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            
            # Determine target based on model type
            model_name_lower = model_name.lower().replace("_", "")
            scalar_models = ["cnnlstm", "pretrainedcnnlstm", "simpleresnet", "physicscnnlstm", "bayesian"]
            is_scalar_model = any(m in model_name_lower for m in scalar_models) and "spatial" not in model_name_lower
            
            if is_scalar_model:
                if scalars is not None:
                     # Use the exact 4-sensor scalars
                     labels = scalars
                     
                     # Fix: Handle dimension mismatch for models expecting single frame output
                     # Models: CNNLSTM, PretrainedCNNLSTM, SimpleResNet, BayesianResNet output (B, 4)
                     # PhysicsCNNLSTM outputs (B, T, 4)
                     if "physicscnnlstm" not in model_name_lower and labels.dim() == 3:
                         labels = labels[:, -1, :] # Take last time step -> (B, 4)
                else:
                    # Fallback to heatmap max if scalars absent (should not happen with new dataset)
                    if labels_raw.dim() == 5: 
                        labels = labels_raw.amax(dim=(2, 3, 4))
                    else:
                        labels = labels_raw.amax(dim=(1, 2, 3))
            else:
                # Spatial models use the heatmap
                labels = labels_raw

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if mask is not None:
                mask = mask.to(device, non_blocking=True)
            
            # Extract Optical Flow if using AdvancedBioHeatLoss (5-channel input)
            flow = None
            if isinstance(criterion_instance, PhysicsInformedLoss) and images.shape[2] == 5:
                # images is (B, T, C, H, W). Last 2 channels are flow
                flow = images[:, :, 3:, :, :]
            
            optimizer_instance.zero_grad()
            
            if scaler is not None:
                with autocast('cuda'):
                    outputs = model_instance(images)
                    
                    # Handle multiple outputs
                    alpha_map, beta_map = None, None
                    kl_loss = 0.0
                    if isinstance(outputs, tuple):
                        if len(outputs) == 3:
                            outputs, alpha_map, beta_map = outputs
                        elif len(outputs) == 2:
                            # Only treat the second element as KL divergence when
                            # it's a 0-D scalar (Bayesian convention).  Models like
                            # ConvLTC return (output, hidden_state) — we discard
                            # the hidden state here.
                            outputs, aux = outputs
                            if isinstance(aux, torch.Tensor) and aux.dim() == 0:
                                kl_loss = aux
                        
                    # Handle physics model output
                    if isinstance(criterion_instance, AdvancedBioHeatLoss):
                        labels_phys = _pool_target_for_bioheat(
                            labels.float(), criterion_instance
                        )
                        loss = criterion_instance(outputs, labels_phys, flow=flow, mask=mask,
                                                 alpha_map=alpha_map, beta_map=beta_map)
                    elif isinstance(criterion_instance, PhysicsInformedLoss):
                        if outputs.dim() == 3 and outputs.shape[1:] == (4, 4):
                            outputs = outputs.mean(dim=(1, 2))
                        loss = criterion_instance(outputs, labels.float(), mask=mask)
                    else:
                        outputs, target = align_outputs_target(outputs, labels.float())
                        loss = criterion_instance(outputs, target)
                    
                    if isinstance(kl_loss, torch.Tensor) or kl_loss > 0:
                        loss = loss + 0.1 * kl_loss # Default weight 0.1 for generic training
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer_instance)
                torch.nn.utils.clip_grad_norm_(model_instance.parameters(), max_norm=1.0)
                
                scaler.step(optimizer_instance)
                scaler.update()
            else:
                outputs = model_instance(images)
                
                # Handle multiple outputs
                alpha_map, beta_map = None, None
                kl_loss = 0.0
                if isinstance(outputs, tuple):
                    if len(outputs) == 3:
                        outputs, alpha_map, beta_map = outputs
                    elif len(outputs) == 2:
                        outputs, aux = outputs
                        if isinstance(aux, torch.Tensor) and aux.dim() == 0:
                            kl_loss = aux
                    
                if isinstance(criterion_instance, AdvancedBioHeatLoss):
                    labels_phys = _pool_target_for_bioheat(
                        labels.float(), criterion_instance
                    )
                    loss = criterion_instance(outputs, labels_phys, flow=flow, mask=mask,
                                             alpha_map=alpha_map, beta_map=beta_map)
                elif isinstance(criterion_instance, PhysicsInformedLoss):
                    if outputs.dim() == 3 and outputs.shape[1:] == (4, 4):
                        outputs = outputs.mean(dim=(1, 2))
                    loss = criterion_instance(outputs, labels.float(), mask=mask)
                else:
                        outputs, target = align_outputs_target(outputs, labels.float())
                        loss = criterion_instance(outputs, target)
                        
                if isinstance(kl_loss, torch.Tensor) or kl_loss > 0:
                    loss = loss + 0.1 * kl_loss
                
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
            for batch in val_progress:
                scalars = None
                # Robust unpacking of batch
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 4:
                        images, labels_raw, mask, scalars = batch[0], batch[1], batch[2], batch[3]
                        if isinstance(mask, torch.Tensor) and mask.dim() != 4:
                            mask = None
                    elif len(batch) == 3:
                         images, labels_raw, mask = batch[0], batch[1], batch[2]
                         if isinstance(mask, torch.Tensor) and mask.dim() != 4:
                             mask = None
                    else:
                        images, labels = batch[0], batch[1]
                        mask = None
                        labels_raw = labels
                
                # Determine target based on model type (same logic as the training pass).
                model_name_lower = model_name.lower().replace("_", "")
                scalar_models = [
                    "cnnlstm", "pretrainedcnnlstm", "simpleresnet",
                    "physicscnnlstm", "bayesian",
                ]
                is_scalar_model = (
                    any(m in model_name_lower for m in scalar_models)
                    and "spatial" not in model_name_lower
                )

                if is_scalar_model:
                    if scalars is not None:
                         labels = scalars
                         # Fix: Handle dimension mismatch for models expecting single frame output
                         if "physicscnnlstm" not in model_name_lower and labels.dim() == 3:
                             labels = labels[:, -1, :] # Take last time step -> (B, 4)
                    else:
                        if labels_raw.dim() == 5: 
                            labels = labels_raw.amax(dim=(2, 3, 4))
                        else:
                            labels = labels_raw.amax(dim=(1, 2, 3))
                else:
                    labels = labels_raw
                    
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                if mask is not None:
                    mask = mask.to(device, non_blocking=True)
                
                if scaler is not None:
                    with autocast('cuda'):
                        outputs = model_instance(images)
                        
                        # Handle multiple outputs (Spatial Map + Physics Params)
                        alpha_map, beta_map = None, None
                        if isinstance(outputs, tuple):
                            if len(outputs) == 3:
                                outputs, alpha_map, beta_map = outputs
                            elif len(outputs) == 2:
                                outputs, _aux = outputs
                            
                        if isinstance(criterion_instance, AdvancedBioHeatLoss):
                            # Extract flow if present (channels 3 and 4)
                            flow = images[:, :, 3:, :, :] if images.shape[2] >= 5 else None
                            labels_phys = _pool_target_for_bioheat(
                                labels.float(), criterion_instance
                            )
                            loss = criterion_instance(outputs, labels_phys, flow=flow, mask=mask, 
                                                     alpha_map=alpha_map, beta_map=beta_map)
                        elif isinstance(criterion_instance, PhysicsInformedLoss):
                            if outputs.dim() == 3 and outputs.shape[1:] == (4, 4):
                                outputs = outputs.mean(dim=(1, 2))
                            loss = criterion_instance(outputs, labels.float(), mask=mask)
                        else:
                            outputs, target = align_outputs_target(outputs, labels.float())
                            loss = criterion_instance(outputs, target)
                else:
                    outputs = model_instance(images)
                    
                    # Handle multiple outputs (Spatial Map + Physics Params)
                    alpha_map, beta_map = None, None
                    if isinstance(outputs, tuple):
                        if len(outputs) == 3:
                            outputs, alpha_map, beta_map = outputs
                        elif len(outputs) == 2:
                            outputs, _aux = outputs
                        
                    if isinstance(criterion_instance, AdvancedBioHeatLoss):
                        # Extract flow if present (channels 3 and 4)
                        flow = images[:, :, 3:, :, :] if images.shape[2] >= 5 else None
                        labels_phys = _pool_target_for_bioheat(
                            labels.float(), criterion_instance
                        )
                        loss = criterion_instance(outputs, labels_phys, flow=flow, mask=mask,
                                                 alpha_map=alpha_map, beta_map=beta_map)
                    elif isinstance(criterion_instance, PhysicsInformedLoss):
                        loss = criterion_instance(outputs, labels.float(), mask=mask)
                    else:
                        outputs, target = align_outputs_target(outputs, labels.float())
                        loss = criterion_instance(outputs, target)
                
                val_loss += loss.item()
                val_progress.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to WandB (guarded — only when a run is already active)
        if getattr(wandb, "run", None) is not None:
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
                       choices=['cnnlstm', 'pretrained_cnnlstm', 'simple_resnet', 'physics_cnnlstm', 'spatial_physics_cnnlstm', 'all'],
                       default=['all'], help="Models to train")
    parser.add_argument("--masked", action="store_true", help="Enable thermometer artifact masking")
    args = parser.parse_args()
    
    # Model parameters - optimized for speed
    # 3 RGB channels + 2 Optical Flow channels = 5 channels
    frame_shape = (64, 64, 5)
    time_steps = 3
    
    # Ensure masked directory exists
    if args.masked:
        os.makedirs("models/masked", exist_ok=True)
        os.makedirs("checkpoints/masked", exist_ok=True)
    
    print(f"=== Preparing Dataset (Masked: {args.masked}) ===")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        sequence_length=time_steps,
        transform=transform,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=args.masked,
        use_physics_prior=False
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
        models_to_train = [
            'cnnlstm', 
            'pretrained_cnnlstm', 
            'simple_resnet', 
            'physics_cnnlstm',
            'spatial_physics_cnnlstm'
        ]
    else:
        models_to_train = args.models
    
    # Initialize WandB
    run_name = "standard-models-benchmark"
    if args.masked:
        run_name += "-masked"
    wandb.init(project="video-temperature-regression", name=run_name, config=args)
    
    # Training results
    results = {}
    
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    if args.masked:
        os.makedirs("models/masked", exist_ok=True)
    
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training {model_type.upper()}")
        print(f"{'='*50}")

        # Check if we should skip this model
        if args.masked:
            model_path = f"models/masked/{model_type}_model.pth"
            save_path = f"models/masked/{model_type}_model.pth"
        else:
            model_path = f"models/{model_type}_model.pth"
            save_path = f"models/{model_type}_model.pth"
        
        should_rerun = config["force_rerun"].get(model_type, False)
        
        # Force rerun if requested via args (implicit in this script usage)
        if os.path.exists(model_path) and not should_rerun:
            print(f"Model {model_type} already exists at {model_path} and force_rerun is False. Skipping.")
            continue
        
        # Create model
        model = create_model(model_type, frame_shape, time_steps)
        
        # Create criterion and optimizer
        if model_type == "physics_cnnlstm":
            # Use PhysicsInformedLoss for the basic physics model
            criterion = PhysicsInformedLoss(physics_weight=0.1)
        elif model_type == "spatial_physics_cnnlstm":
            # Use AdvancedBioHeatLoss for the spatial physics model (Issue #41, #42)
            criterion = AdvancedBioHeatLoss(
                physics_weight=0.1, 
                spatial_params=True, 
                learnable_params=True,
                frame_shape=(4, 4) # Matches SpatialPhysicsCNNLSTM output
            )
        else:
            criterion = nn.MSELoss()
            
        # Ensure criterion is on device (crucial for learnable physics parameters)
        criterion = criterion.to(device)
            
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
                patience=args.patience,
                model_save_path=save_path,
                masked=args.masked
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
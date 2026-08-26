"""
Leave-One-Sequence-Out (LOSO) Cross-Validation Script.

This script performs rigorous validation by training on (N-1) sequences 
and testing on the held-out sequence. This proves the spatial and 
setup generalization of our temperature regression models.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

try:
    import wandb
except ImportError:  # wandb is optional for the LOSO driver
    wandb = None
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet, SpatialResNet
from models.bayesian import (
    BayesianCNNLSTM,
    BayesianResNet,
    BayesianSpatialResNet,
    FullBayesianResNet,
)
from models.conv_ltc import ConvLTC
from models.kan import KANResNet, SpatialKANBioheat
from physics.bioheat_loss import AdvancedBioHeatLoss
from physics.loss import PhysicsInformedLoss
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from training.train_all_models import align_outputs_target, train_model_with_validation
from utils.augmentation import AugmentedSequenceSubset
from utils.seed_utils import seed_worker, set_global_seed
from utils.sequence_dataset import SequenceHeatmapDataset


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_run_config(save_path, holdout_seq, model_type, args, seed):
    config = {
        "model": model_type,
        "fold": holdout_seq,
        "seed": seed,
        "weight_decay": args.weight_decay,
        "augment": args.augment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "masked": args.masked,
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hostname": platform.node(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config_path = save_path.replace(".pth", "_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def run_loso_fold(holdout_seq, model_type, args):
    """Run a single LOSO fold holding out holdout_seq."""
    print(f"\n{'='*60}")
    print(f"FOLD: Holding out {holdout_seq}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load full dataset
    full_dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        sequence_length=5,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=args.masked,
    )

    # Split based on sequence names
    train_indices = []
    test_indices = []

    # SequenceHeatmapDataset uses .videos list for meta, but indices map to (vid_idx, start)
    # We match by video basename (without extension) so the holdout id can be a
    # legacy 'sequence_X' tag OR a real filename like 'US_001_30W_10min'.
    holdout_token = str(holdout_seq).replace(".mp4", "")
    for idx_in_ds, (vid_idx, start_frame) in enumerate(full_dataset.indices):
        vid_path = full_dataset.videos[vid_idx]["path"]
        vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
        if holdout_token == vid_basename or holdout_token in vid_path:
            test_indices.append(idx_in_ds)
        else:
            train_indices.append(idx_in_ds)

    if not test_indices:
        print(f"Warning: No samples found for sequence {holdout_seq}. Skipping fold.")
        return None

    # Further split train into train/val (85/15). Seeded so the split is
    # reproducible across runs/seeds instead of relying on global RNG state.
    split_rng = np.random.RandomState(args.seed)
    split_rng.shuffle(train_indices)
    split = int(0.85 * len(train_indices))
    val_indices = train_indices[split:]
    train_indices = train_indices[:split]

    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_ds = Subset(full_dataset, test_indices)

    if args.augment:
        train_ds = AugmentedSequenceSubset(train_ds)

    print(f"Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    frame_shape = (64, 64, 5)
    time_steps = 5

    if model_type == "ConvectionBioheat":
        # The ultimate spatial physics model (Issue #41, #42)
        model = SpatialPhysicsCNNLSTM(
            frame_shape=frame_shape, time_steps=time_steps, pretrained=True
        )
        criterion = AdvancedBioHeatLoss(
            physics_weight=0.1,
            spatial_params=True,
            learnable_params=True,
            frame_shape=(4, 4),
        )
    elif model_type == "SpatialPhysicsCNNLSTM":
        model = SpatialPhysicsCNNLSTM(
            frame_shape=frame_shape, time_steps=time_steps, pretrained=True
        )
        criterion = AdvancedBioHeatLoss(
            physics_weight=0.1, spatial_params=True, learnable_params=True
        )
    elif model_type == "PhysicsCNNLSTM":
        model = PhysicsCNNLSTM(
            frame_shape=frame_shape, time_steps=time_steps, pretrained=True
        )
        criterion = PhysicsInformedLoss(physics_weight=0.1)
    elif model_type == "CNNLSTM":
        model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
        criterion = torch.nn.MSELoss()
    elif model_type == "PretrainedCNNLSTM":
        # We need a dummy backbone for the constructor in backbones.py
        from torchvision.models import resnet18

        backbone = resnet18()
        backbone.fc = torch.nn.Linear(512, 1)
        model = PretrainedCNNLSTM(
            backbone, frame_shape=frame_shape, time_steps=time_steps
        )
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
    elif model_type == "KANResNet":
        model = KANResNet(frame_shape=frame_shape)
        criterion = torch.nn.MSELoss()
    elif model_type == "SpatialKANBioheat":
        model = SpatialKANBioheat(
            frame_shape=frame_shape, time_steps=time_steps, output_hw=(4, 4)
        )
        criterion = AdvancedBioHeatLoss(
            physics_weight=0.1,
            spatial_params=True,
            learnable_params=True,
            frame_shape=(4, 4),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Ensure criterion is on device
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=args.weight_decay
    )

    # Path for fold checkpoint
    os.makedirs(f"checkpoints/loso/{model_type}", exist_ok=True)
    save_path = f"checkpoints/loso/{model_type}/fold_{holdout_seq}.pth"
    _write_run_config(save_path, holdout_seq, model_type, args, args.seed)

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
        masked=args.masked,
    )

    # Evaluate on held-out sequence
    model.load_state_dict(torch.load(save_path))
    model.eval()

    is_bayesian = "Bayesian" in model_type
    mc_samples = 10 if is_bayesian else 1

    # Spatial models emit a temperature field; we evaluate it against the
    # pooled ground-truth heatmap in addition to the scalar reduction.
    spatial_models = {
        "SpatialResNet",
        "ConvectionBioheat",
        "SpatialPhysicsCNNLSTM",
        "ConvLTC",
        "SpatialKANBioheat",
    }
    is_spatial = model_type in spatial_models

    errors = []
    field_abs_errors = []
    field_sq_errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Fold"):
            scalars = None
            gt_maps = None
            if len(batch) == 4:
                imgs, gt_maps, mask, scalars = batch
                gt = scalars
            elif len(batch) == 3:
                imgs, gt, mask = batch
            else:
                imgs, gt = batch
                mask = None

            if args.masked and mask is not None:
                # imgs is (B, T, C, H, W). mask can be (B, 1, H, W),
                # (B, T, 1, H, W) or (B, T, H, W).
                if mask.dim() == 4 and imgs.dim() == 5:
                    mask = mask.unsqueeze(1)  # -> (B, 1, 1, H, W)
                elif mask.dim() == 4 and imgs.dim() == 4:
                    pass  # (B, 1, H, W) ok
                # else assume already broadcast-compatible
                imgs = imgs * (1.0 - mask.float())

            imgs = imgs.to(device)

            # Monte Carlo sampling for Bayesian models
            batch_samples = []
            batch_field_samples = []
            for _ in range(mc_samples):
                out = model(imgs)

                # Handle multiple outputs (Issue #41)
                if isinstance(out, tuple):
                    out = out[0]

                # Keep a copy of the raw field BEFORE scalar reduction so
                # we can evaluate per-pixel temperature estimation.
                if is_spatial and gt_maps is not None:
                    batch_field_samples.append(out.detach().cpu())

                # Scalar reduction (sensor-level metric).
                gt_t = gt if torch.is_tensor(gt) else torch.as_tensor(gt)
                out_red, _ = align_outputs_target(out, gt_t.to(out.device).float())
                batch_samples.append(out_red.cpu().numpy())

            # Mean across MC samples (now in helper-reduced shape).
            out_mean = np.mean(batch_samples, axis=0)

            # Pair the reduced out_mean with the gt through the helper once
            # more so both come out length-matched (the helper truncates to
            # the smaller of the two as a last resort).
            out_t = torch.as_tensor(out_mean)
            gt_t = gt if torch.is_tensor(gt) else torch.as_tensor(gt)
            out_aligned, gt_aligned = align_outputs_target(out_t, gt_t.float())

            batch_errors = out_aligned.cpu().numpy().reshape(
                -1
            ) - gt_aligned.cpu().numpy().reshape(-1)
            errors.extend(batch_errors.tolist())

            # --- Field-level temperature estimation ---
            if is_spatial and batch_field_samples:
                # Stack MC samples on a new leading axis then mean.
                fields = torch.stack(batch_field_samples, dim=0).mean(dim=0)
                # fields is the model output; could be (B, T, h, w),
                # (B, T, C, h, w), or (B, h, w).  Reduce to (B, T, h, w)
                # or (B, h, w) by averaging any channel axis.
                if fields.dim() == 5:
                    fields = fields.mean(dim=2)  # (B, T, h, w)
                elif fields.dim() == 4 and fields.shape[1] not in (1, imgs.shape[1]):
                    # (B, C, h, w) without time -> mean over C
                    fields = fields.mean(dim=1)  # (B, h, w)

                # Pool the ground-truth heatmap (B, T, 1, H, W) down to the
                # predicted spatial resolution and align temporal shape.
                gt_field = gt_maps.float()  # (B, T, 1, H, W)
                if gt_field.dim() == 5:
                    gt_field = gt_field.squeeze(2)  # (B, T, H, W)
                if fields.dim() == 3 and gt_field.dim() == 4:
                    gt_field = gt_field[:, -1]  # (B, H, W)
                if (
                    fields.dim() == 4
                    and gt_field.dim() == 4
                    and fields.shape[1] != gt_field.shape[1]
                ):
                    # mismatched T -> use last frame on both
                    fields = fields[:, -1:]
                    gt_field = gt_field[:, -1:]

                # Adaptive-pool ground truth to predicted spatial shape.
                target_hw = fields.shape[-2:]
                if gt_field.shape[-2:] != target_hw:
                    flat = gt_field.reshape(-1, 1, *gt_field.shape[-2:])
                    pooled = torch.nn.functional.adaptive_avg_pool2d(flat, target_hw)
                    gt_field = pooled.reshape(*gt_field.shape[:-2], *target_hw)

                diff = (fields - gt_field).numpy().reshape(-1)
                field_abs_errors.extend(np.abs(diff).tolist())
                field_sq_errors.extend((diff**2).tolist())

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))

    field_mae = float(np.mean(field_abs_errors)) if field_abs_errors else float("nan")
    field_rmse = (
        float(np.sqrt(np.mean(field_sq_errors))) if field_sq_errors else float("nan")
    )

    if is_spatial and field_abs_errors:
        print(
            f"Fold {holdout_seq} Results: MAE={mae:.4f} K, RMSE={rmse:.4f} K | "
            f"Field MAE={field_mae:.4f} K, Field RMSE={field_rmse:.4f} K"
        )
    else:
        print(f"Fold {holdout_seq} Results: MAE={mae:.4f} K, RMSE={rmse:.4f} K")

    return {
        "fold": holdout_seq,
        "mae": float(mae),
        "rmse": float(rmse),
        "field_mae": field_mae,
        "field_rmse": field_rmse,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ConvectionBioheat")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/level1_cropped",
        help="Where to scan for fold ids (one fold per video file).",
    )
    parser.add_argument(
        "--folds",
        type=str,
        default=None,
        help="Comma-separated list of fold ids to run. Default: every .mp4 in --data_dir.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging entirely (overrides --wandb_project).",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="video-regression-loso",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global RNG seed for reproducibility."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (L2 regularization).",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply random flip + brightness jitter to the training split.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Identify sequences: one fold per video in the canonical dataset directory.
    if args.folds:
        seq_ids = [s.strip() for s in args.folds.split(",") if s.strip()]
    else:
        import glob as _glob

        seq_ids = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in _glob.glob(os.path.join(args.data_dir, "*.mp4"))
        )
    if not seq_ids:
        raise SystemExit(
            f"No fold ids resolved. Pass --folds or populate {args.data_dir} with .mp4 videos."
        )

    if wandb is not None and not args.no_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                name=f"LOSO_{args.model}{'_masked' if args.masked else ''}",
                config=vars(args),
                reinit=True,
            )
        except Exception as e:  # offline / no auth
            print(f"[loso] wandb init failed ({e}); continuing without logging.")

    results = []
    for seq_id in seq_ids:
        try:
            res = run_loso_fold(seq_id, args.model, args)
            if res:
                res["model"] = args.model
                results.append(res)
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.log(
                        {
                            f"fold/{seq_id}/mae": res["mae"],
                            f"fold/{seq_id}/rmse": res["rmse"],
                        }
                    )
        except Exception as e:
            print(f"Error in fold {seq_id}: {e}")

    # Aggregate
    if not results:
        print("No results collected.")
        return

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    print(f"MEAN MAE:  {df['mae'].mean():.4f} +/- {df['mae'].std():.4f} K")
    print(f"MEAN RMSE: {df['rmse'].mean():.4f} +/- {df['rmse'].std():.4f} K")

    os.makedirs("results", exist_ok=True)
    tag = (
        f"seed{args.seed}"
        + ("_aug" if args.augment else "")
        + ("_masked" if args.masked else "_unmasked")
    )
    output_path = f"results/loso_{args.model}_{tag}.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

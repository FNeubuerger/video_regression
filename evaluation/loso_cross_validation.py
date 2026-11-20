"""
Leave-One-Sequence-Out (LOSO) Cross-Validation Script.

Evaluate held-out phantom sequences with strict probe-based scoring.
CNNLSTM and SimpleResNet retraining use measured probes and sequence validation.
"""

import argparse
import copy
import json
import os
import platform
import re
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
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.loso_metrics import batch_errors
from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet, SpatialResNet
from models.bayesian import BayesianCNNLSTM, BayesianResNet, FullBayesianResNet
from models.conv_ltc import ConvLTC
from models.kan import KANResNet, SpatialKANBioheat
from physics.bioheat_loss import AdvancedBioHeatLoss
from physics.loss import PhysicsInformedLoss
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from training.loso_probe_training import fit_probe_model, split_by_sequence
from training.train_all_models import train_model_with_validation
from utils.augmentation import AugmentedSequenceSubset
from utils.seed_utils import seed_worker, set_global_seed
from utils.sequence_dataset import SequenceHeatmapDataset

PROBE_TRAINING_MODELS = {"CNNLSTM", "SimpleResNet"}


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


def _write_run_config(
    save_path, holdout_seq, model_type, args, protocol, val_sequences
):
    config = {
        "model": model_type,
        "fold": holdout_seq,
        "seed": args.seed,
        "weight_decay": args.weight_decay,
        "augment": args.augment,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "masked": args.masked,
        "training_protocol": protocol,
        "validation_sequences": val_sequences,
        "max_fit_windows": args.max_fit_windows,
        "max_test_windows": args.max_test_windows,
        "data_dir": args.data_dir,
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


def _sample_windows(indices, limit):
    if limit is None or len(indices) <= limit:
        return indices
    selected = np.linspace(0, len(indices) - 1, limit, dtype=int)
    return [indices[i] for i in selected]


def run_loso_fold(holdout_seq, model_type, args):
    """Run a single LOSO fold holding out holdout_seq."""
    print(f"\n{'='*60}")
    print(f"FOLD: Holding out {holdout_seq}")
    print(f"{'='*60}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    set_global_seed(args.seed)
    evaluation_only = bool(args.evaluation_checkpoint)
    probe_training = not evaluation_only and model_type in PROBE_TRAINING_MODELS
    protocol = "existing_checkpoint" if evaluation_only else "legacy"
    if probe_training:
        protocol = "measured_probe_sequence_val"

    # Load full dataset
    full_dataset = SequenceHeatmapDataset(
        data_dir=args.data_dir,
        raw_dir="data/level0_raw",
        sequence_length=5,
        target_size=(64, 64),
        use_optical_flow=True,
        use_artifact_masking=args.masked,
        return_probe_mask=probe_training,
    )

    # Split based on sequence names
    fit_windows = []
    test_indices = []

    # SequenceHeatmapDataset uses .videos list for meta, but indices map to (vid_idx, start)
    # Match exact video basenames to avoid unintentionally holding out other runs.
    holdout_token = str(holdout_seq).removesuffix(".mp4")
    for idx_in_ds, (vid_idx, _) in enumerate(full_dataset.indices):
        vid_path = full_dataset.videos[vid_idx]["path"]
        vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
        if holdout_token == vid_basename:
            test_indices.append(idx_in_ds)
        else:
            fit_windows.append((vid_basename, idx_in_ds))

    if not test_indices:
        raise ValueError(f"No samples found for sequence {holdout_seq}")

    val_sequences = []
    if probe_training:
        train_indices, val_indices, val_sequences = split_by_sequence(
            fit_windows, args.seed
        )
    else:
        train_indices = [i for _, i in fit_windows]
        np.random.RandomState(args.seed).shuffle(train_indices)
        split = int(0.85 * len(train_indices))
        train_indices, val_indices = train_indices[:split], train_indices[split:]

    train_indices = _sample_windows(train_indices, args.max_fit_windows)
    val_indices = _sample_windows(val_indices, args.max_fit_windows)
    train_ds = Subset(full_dataset, train_indices)
    val_ds = Subset(full_dataset, val_indices)
    test_dataset = copy.copy(full_dataset)
    test_dataset.return_probe_mask = True
    available_windows = len(test_indices)
    test_indices = _sample_windows(test_indices, args.max_test_windows)
    test_ds = Subset(test_dataset, test_indices)

    if args.augment:
        train_ds = AugmentedSequenceSubset(train_ds)

    print(f"Split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    if val_sequences:
        print(f"Validation sequences: {', '.join(val_sequences)}")

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
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Create model
    frame_shape = (64, 64, 5)
    time_steps = 5

    if model_type == "ConvectionBioheat":
        # The ultimate spatial physics model (Issue #41, #42)
        model = SpatialPhysicsCNNLSTM(
            frame_shape=frame_shape,
            time_steps=time_steps,
            pretrained=args.evaluation_checkpoint is None,
        )
        criterion = AdvancedBioHeatLoss(
            physics_weight=0.1,
            spatial_params=True,
            learnable_params=True,
            frame_shape=(4, 4),
        )
    elif model_type == "SpatialPhysicsCNNLSTM":
        model = SpatialPhysicsCNNLSTM(
            frame_shape=frame_shape,
            time_steps=time_steps,
            pretrained=args.evaluation_checkpoint is None,
        )
        criterion = AdvancedBioHeatLoss(
            physics_weight=0.1, spatial_params=True, learnable_params=True
        )
    elif model_type == "PhysicsCNNLSTM":
        model = PhysicsCNNLSTM(
            frame_shape=frame_shape,
            time_steps=time_steps,
            pretrained=args.evaluation_checkpoint is None,
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

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=args.weight_decay
    )

    # Path for fold checkpoint
    save_path = args.evaluation_checkpoint
    best_epoch = val_mse = None
    if save_path is None:
        checkpoint_dir = os.path.join("checkpoints", "loso_corrected", args.run_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"fold_{holdout_token}.pth")
        if os.path.exists(save_path):
            raise FileExistsError(save_path)
        _write_run_config(
            save_path, holdout_seq, model_type, args, protocol, val_sequences
        )
        if probe_training:
            best_epoch, val_mse = fit_probe_model(
                model,
                train_loader,
                val_loader,
                optimizer,
                device,
                args.epochs,
                save_path,
            )
        else:
            train_model_with_validation(
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
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=True))
    model.to(device)
    model.eval()

    is_bayesian = "Bayesian" in model_type
    mc_samples = 10 if is_bayesian else 1

    # Spatial models are scored at measured probe locations only.
    spatial_models = {
        "SpatialResNet",
        "ConvectionBioheat",
        "SpatialPhysicsCNNLSTM",
        "ConvLTC",
        "SpatialKANBioheat",
    }
    is_spatial = model_type in spatial_models

    errors = []
    partial_errors = []
    probe_errors = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Fold"):
            scalar, partial, probes = batch_errors(
                model, batch, device, spatial=is_spatial, mc_samples=mc_samples
            )
            errors.extend(scalar.tolist())
            partial_errors.extend(partial.tolist())
            probe_errors.extend(probes.tolist())

    if not errors:
        print(f"[loso] No scorable four-probe windows on {holdout_seq}")

    return {
        "fold": holdout_seq,
        "mae": np.mean(np.abs(errors)) if errors else None,
        "rmse": np.sqrt(np.mean(np.square(errors))) if errors else None,
        "partial_mean_mae": (
            np.mean(np.abs(partial_errors)) if partial_errors else None
        ),
        "probe_mae": np.mean(np.abs(probe_errors)) if probe_errors else None,
        "probe_rmse": (
            np.sqrt(np.mean(np.square(probe_errors))) if probe_errors else None
        ),
        "available_windows": available_windows,
        "sampled_windows": len(test_ds),
        "scored_windows": len(errors),
        "partial_windows": len(partial_errors),
        "scored_probes": len(probe_errors),
        "checkpoint": save_path,
        "masked": args.masked,
        "evaluation_only": evaluation_only,
        "training_protocol": protocol,
        "train_windows": 0 if evaluation_only else len(train_ds),
        "validation_windows": 0 if evaluation_only else len(val_ds),
        "validation_sequences": ";".join(val_sequences),
        "best_epoch": best_epoch,
        "validation_probe_mse": val_mse,
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ConvectionBioheat")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU.")
    parser.add_argument(
        "--evaluation_checkpoint", help="Score an existing checkpoint for one fold."
    )
    parser.add_argument(
        "--max_test_windows",
        type=int,
        help="Pilot only: cap evenly spaced test windows.",
    )
    parser.add_argument(
        "--run_id", help="Unique name for the isolated result directory."
    )
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
    parser.add_argument(
        "--max_fit_windows",
        type=int,
        help="Pilot only: cap training and validation windows separately.",
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_fit_windows is not None and (
        args.max_fit_windows < 1 or args.evaluation_checkpoint
    ):
        parser.error("--max_fit_windows requires training and a positive window count")
    if args.max_test_windows is not None and args.max_test_windows < 1:
        parser.error("--max_test_windows must be positive")
    if args.run_id is None:
        args.run_id = f"{args.model}-{time.time_ns()}"
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.run_id):
        parser.error("--run_id must consist of letters, digits, underscores or hyphens")

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
    if args.evaluation_checkpoint and len(seq_ids) != 1:
        parser.error("--evaluation_checkpoint requires exactly one --folds value")
    if len(set(seq_ids)) != len(seq_ids):
        parser.error("Duplicate fold IDs")

    output_dir = os.path.join("results", "loso_corrected", args.run_id)
    os.makedirs(output_dir, exist_ok=False)
    if not args.evaluation_checkpoint and args.model not in PROBE_TRAINING_MODELS:
        print(
            "[loso] Using legacy training/validation; only held-out scoring is corrected."
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
        res = run_loso_fold(seq_id, args.model, args)
        res["model"] = args.model
        results.append(res)
        if wandb is not None and getattr(wandb, "run", None) is not None:
            wandb.log(
                {
                    f"fold/{seq_id}/mae": res["mae"],
                    f"fold/{seq_id}/rmse": res["rmse"],
                }
            )

    # Aggregate
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("-" * 60)
    eligible = df.loc[df["scored_windows"] > 0, ["mae", "rmse"]]
    print(f"Four-probe endpoint available on {len(eligible)}/{len(df)} folds")
    if not eligible.empty:
        print(f"Mean four-probe MAE:  {eligible['mae'].mean():.4f} K")
        print(f"Mean four-probe RMSE: {eligible['rmse'].mean():.4f} K")

    output_path = os.path.join(output_dir, "folds.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

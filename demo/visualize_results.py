import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import sys
import argparse
from tqdm import tqdm
import torch.nn.functional as F

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_registry import MODEL_REGISTRY
from utils.sequence_dataset import SequenceHeatmapDataset

def main():
    parser = argparse.ArgumentParser(description="Generate Dashboard Visualization Video")
    parser.add_argument("--model", type=str, default="LatentLTC_UNet", help="Model name")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ltc_unet/best_model.pth", help="Checkpoint path")
    parser.add_argument("--video_index", type=int, default=4, help="Index of video (4 = US_005)")
    parser.add_argument("--output", type=str, default="demo/best_model_dashboard.mp4", help="Output path")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    if args.model not in MODEL_REGISTRY:
        print(f"Error: Model {args.model} not found in registry.")
        return

    ModelClass, kwargs = MODEL_REGISTRY[args.model]
    
    # Adjust kwargs for physical models 
    if args.model in ["SpatialPhysicsCNNLSTM", "ConvectionBioheat", "BioheatPINN", "PhysicsCNNLSTM"]:
        kwargs["frame_shape"] = (64, 64, 5) 
        kwargs["time_steps"] = 16

    model = ModelClass(**kwargs)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using untrained weights.")

    model.to(device)
    model.eval()

    # 2. Dataset
    use_flow = args.model in ["SpatialPhysicsCNNLSTM", "ConvectionBioheat", "BioheatPINN", "PhysicsCNNLSTM"]
    
    dataset = SequenceHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(64, 64),
        sequence_length=16,
        stride=1,
        use_optical_flow=use_flow
    )

    # 3. Setup Layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    writer = FFMpegWriter(fps=10)

    # Find start sequence
    start_seq_idx = 0
    for i, (v_idx, f_idx) in enumerate(dataset.indices):
        if v_idx == args.video_index:
            start_seq_idx = i
            break

    # History for line graphs (per sensor if possible)
    num_sensors = 4
    history_gt = [[] for _ in range(num_sensors)]
    history_pred = [[] for _ in range(num_sensors)]
    sensor_labels = ['M1', 'M2', 'M3', 'M4']

    # Get scales for coordinate mapping
    orig_h, orig_w = dataset.videos[args.video_index]['original_size']
    scale_y = 4 / orig_h # to 4x4
    scale_x = 4 / orig_w

    with writer.saving(fig, args.output, dpi=100):
        print(f"Generating dashboard for sequence starting at {start_seq_idx}...")
        
        for i in tqdm(range(args.frames)):
            idx = start_seq_idx + i
            if idx >= len(dataset): break
            
            inputs, sparse_map, prior_map = dataset[idx]
            input_tensor = inputs.unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_out = model(input_tensor)
                if isinstance(pred_out, tuple):
                    pred_out = pred_out[0]
                
                current_prior = prior_map[-1, 0].cpu().numpy()
                current_gt_sparse = sparse_map[-1, 0].cpu().numpy()
                current_img = inputs[-1, :3].permute(1, 2, 0).numpy()
                
                if pred_out.dim() == 5: # Heatmap model
                    current_pred = pred_out[0, -1, 0].cpu().numpy() + current_prior
                elif pred_out.dim() == 4: # 4x4 model
                    pred_4x4 = pred_out[0, -1] # (4, 4)
                    pred_4x4_np = pred_4x4.cpu().numpy()
                    p_up = F.interpolate(pred_4x4.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='bicubic', align_corners=True)
                    current_pred = p_up[0, 0].cpu().numpy() + current_prior
                else: 
                    val = pred_out[0, -1].item() + np.mean(current_prior)
                    current_pred = np.full((64, 64), val)
            
            # Update histories per sensor
            meta = dataset.videos[args.video_index]
            for s_idx in range(num_sensors):
                label = sensor_labels[s_idx]
                if label in meta['sensor_pos']:
                    # Extract GT
                    v_idx, f_idx = dataset.indices[idx]
                    log_idx = min(int(f_idx * (meta['n_logs'] / meta['n_frames'])), meta['n_logs'] - 1)
                    gt_val = meta['temps'][log_idx][s_idx]
                    history_gt[s_idx].append(gt_val)
                    
                    # Pred: sample
                    pos = meta['sensor_pos'][label]['center']
                    if pred_out.dim() == 4:
                        py, px = int(pos[1] * scale_y), int(pos[0] * scale_x)
                        py = min(max(py, 0), 3)
                        px = min(max(px, 0), 3)
                        p_val = current_prior[int(pos[1]*64/orig_h), int(pos[0]*64/orig_w)]
                        history_pred[s_idx].append(pred_4x4_np[py, px] + p_val)
                    else:
                        py, px = int(pos[1] * 64 / orig_h), int(pos[0] * 64 / orig_w)
                        py = min(max(py, 0), 63)
                        px = min(max(px, 0), 63)
                        history_pred[s_idx].append(current_pred[py, px])
                else:
                    history_gt[s_idx].append(np.nan)
                    history_pred[s_idx].append(np.nan)

            # --- PLOTTING ---
            axes[0].clear()
            img_disp = (current_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            axes[0].imshow(np.clip(img_disp, 0, 1))
            axes[0].set_title("Input Thermal Video")
            axes[0].axis('off')

            axes[1].clear()
            axes[1].imshow(current_prior, vmin=30, vmax=70, cmap='inferno', alpha=0.3)
            y, x = np.where(current_gt_sparse > 0)
            if len(x) > 0:
                axes[1].scatter(x, y, c=current_gt_sparse[y, x], vmin=30, vmax=70, cmap='inferno', edgecolors='white', s=150)
            axes[1].set_title(f"GT Sensors (Frame {i})")
            axes[1].axis('off')

            axes[2].clear()
            axes[2].imshow(current_pred, vmin=30, vmax=70, cmap='inferno')
            axes[2].set_title(f"Predicted Heatmap ({args.model})")
            axes[2].axis('off')

            axes[3].clear()
            colors = ['r', 'g', 'b', 'm']
            for s_idx in range(num_sensors):
                if not np.all(np.isnan(history_gt[s_idx])):
                    axes[3].plot(history_gt[s_idx], color=colors[s_idx], linestyle='-', alpha=0.5, label=f'{sensor_labels[s_idx]} GT')
                    axes[3].plot(history_pred[s_idx], color=colors[s_idx], linestyle='--', label=f'{sensor_labels[s_idx]} Pred')
            
            axes[3].set_xlim(0, args.frames)
            axes[3].set_ylim(20, 80)
            axes[3].set_ylabel("Temp (°C)")
            axes[3].set_xlabel("Frame")
            axes[3].legend(loc='upper right', fontsize='small', ncol=2)
            axes[3].grid(True)

            writer.grab_frame()

    print(f"Finished. Saved to {args.output}")

if __name__ == "__main__":
    main()

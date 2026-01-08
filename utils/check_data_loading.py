import sys
import os
import torch
import matplotlib.pyplot as plt

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.heatmap_dataset import TemperatureHeatmapDataset

def main():
    print("Testing TemperatureHeatmapDataset...")
    ds = TemperatureHeatmapDataset(
        data_dir="data/level1_cropped",
        raw_dir="data/level0_raw",
        target_size=(256, 256)
    )
    
    if len(ds) == 0:
        print("Dataset is empty!")
        return

    print(f"Dataset length: {len(ds)}")
    
    # Get a sample
    idx = 100 # Random index
    frame, target, mask, vec = ds[idx]
    
    print(f"Frame shape: {frame.shape}")
    print(f"Target map shape: {target.shape}")
    print(f"Mask map shape: {mask.shape}")
    print(f"Temp Vector: {vec}")
    
    # Visualize
    # Unnormalize frame
    inv_norm = lambda t: t * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    vis_frame = inv_norm(frame).permute(1, 2, 0).numpy()
    # Clip to valid range before casting
    vis_frame = vis_frame.clip(0, 1)
    vis_frame = (vis_frame * 255).astype('uint8')
    
    mask_np = mask[0].numpy()
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(vis_frame)
    plt.title("Input Frame (Resized)")
    
    # Overlay sensors
    ys, xs = mask_np.nonzero()
    plt.scatter(xs, ys, c='red', s=20, label='Sensors')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.imshow(target.squeeze().numpy(), cmap='jet')
    plt.title("Sparse Target Map")
    plt.colorbar()
    
    out_path = "data/level1_cropped/dataset_check.png"
    plt.savefig(out_path)
    print(f"Saved check image to {out_path}")

if __name__ == "__main__":
    main()

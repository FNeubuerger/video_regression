"""
Visualization of Advection: Temperature Gradients and Optical Flow.
This script demonstrates how physical motion (Optical Flow) interacts 
with the temperature field (Advection term v . grad T).
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from torchvision import transforms

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.dataset import TemperatureSequenceDataset
from physics.models import SpatialPhysicsCNNLSTM

def plot_advection(checkpoint_path, data_dir="data", output_dir="results/plots/advection"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    frame_shape = (64, 64, 5)
    time_steps = 5
    model = SpatialPhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Load Dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = TemperatureSequenceDataset(
        data_dir=data_dir,
        sequence_length=time_steps,
        transform=transform,
        use_optical_flow=True,
        image_size=(64, 64)
    )
    
    # 3. Process a few samples with high motion
    indices = [500, 1500, 3500]
    
    for idx in indices:
        if idx >= len(dataset): continue
        
        # imgs: (T, 5, 64, 64) -> [R, G, B, flow_x, flow_y]
        imgs, target, _ = dataset[idx]
        imgs_batch = imgs.unsqueeze(0).to(device)
        
        with torch.no_grad():
            temp_map, _, _ = model(imgs_batch)
            
        # Get last frame's predicted temp map (4x4)
        T_last = temp_map[0, -1].cpu().numpy()
        
        # Get last frame's optical flow from dataset
        # flow is in channels 3 and 4 of the 'imgs' tensor
        last_flow = imgs[-1, 3:5].cpu().numpy() # (2, 64, 64)
        vx = last_flow[0]
        vy = last_flow[1]
        
        # Compute Gradients of predicted Temperature map
        # T_last is (4, 4)
        grad_Ty, grad_Tx = np.gradient(T_last)
        
        # Downsample flow to match 4x4
        # We'll use simple averaging for visualization
        vx_small = np.zeros((4,4))
        vy_small = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                vx_small[i,j] = vx[i*16:(i+1)*16, j*16:(j+1)*16].mean()
                vy_small[i,j] = vy[i*16:(i+1)*16, j*16:(j+1)*16].mean()

        # Final Advection Term: vx*Tx + vy*Ty
        advection = vx_small * grad_Tx + vy_small * grad_Ty
        
        # Plotting
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. Temperature Map
        im1 = axes[0].imshow(T_last, cmap='hot', interpolation='nearest')
        axes[0].set_title("Temperature $T/K$")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Velocity Field (Quiver)
        Y, X = np.mgrid[0:4, 0:4]
        axes[1].quiver(X, Y, vx_small, -vy_small, color='blue') # Negative vy because image y is down
        axes[1].set_title("Flow Field $\mathbf{v}$")
        axes[1].set_xlim(-0.5, 3.5)
        axes[1].set_ylim(3.5, -0.5)
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        
        # 3. Temperature Gradient (Quiver)
        axes[2].quiver(X, Y, grad_Tx, -grad_Ty, color='red')
        axes[2].set_title("$\nabla T$ Gradient")
        axes[2].set_xlim(-0.5, 3.5)
        axes[2].set_ylim(3.5, -0.5)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")

        # 4. Advection Term Map
        im4 = axes[3].imshow(advection, cmap='RdBu_r', interpolation='nearest')
        axes[3].set_title("Advection $\mathbf{v} \cdot \\nabla T$")
        axes[3].set_xlabel("x")
        axes[3].set_ylabel("y")
        plt.colorbar(im4, ax=axes[3], label='$K/s$')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/advection_sample_{idx}.png")
        plt.close()
        print(f"Saved advection visualization to {output_dir}/advection_sample_{idx}.png")

if __name__ == "__main__":
    cp = "models/masked/spatial_physics_cnnlstm_model.pth"
    if os.path.exists(cp):
        plot_advection(cp)
    else:
        print(f"Checkpoint {cp} not found.")


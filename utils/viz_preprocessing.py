import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_flow_hsv(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def main():
    # Paths
    base_dir = "/mnt/data2/video_regression/data/sequence_1/"
    frame1_path = os.path.join(base_dir, "frame_50_label_31.6.png")
    frame2_path = os.path.join(base_dir, "frame_51_label_31.65.png")
    output_path = "/mnt/data2/video_regression/paper/figures/preprocessing_pipeline.png"
    
    print(f"Reading frames from {base_dir}...")
    img1 = cv2.imread(frame1_path)
    img2 = cv2.imread(frame2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not read frames. using fallback indices.")
        # Fallback to file listing logic if specific files don't exist
        files = sorted([f for f in os.listdir(base_dir) if f.endswith(".png")])
        if len(files) < 2:
            print("Not enough files in directory.")
            return
        frame1_path = os.path.join(base_dir, files[100])
        frame2_path = os.path.join(base_dir, files[101])
        img1 = cv2.imread(frame1_path)
        img2 = cv2.imread(frame2_path)

    # 1. Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 2. Optical Flow
    print("Computing Optical Flow...")
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # 3. Visualize
    print("Generating Figure...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Raw Frame
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Frame $t$ (RGB)")
    axes[0].axis('off')
    
    # Next Frame
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Input Frame $t+1$ (RGB)")
    axes[1].axis('off')
    
    # Flow
    flow_rgb = visualize_flow_hsv(flow)
    axes[2].imshow(flow_rgb)
    axes[2].set_title("Dense Optical Flow\n(Magnitude & Direction)")
    axes[2].axis('off')

    # Input Tensor Conceptualization
    # We will simulate the 5-channel stack by showing the channels
    stack_viz = np.zeros((128, 128, 3), dtype=np.uint8) # Placeholder
    
    # Better: Show the decomposition
    # R G B Fx Fy
    
    # Let's show the components of the stack
    # We can use a grid spec to show the 5 channels
    axes[3].axis('off')
    axes[3].set_title("5-Channel Input Tensor\n(RGB + Flow X + Flow Y)")
    
    # Create a sub-figure for the stack
    sub_ax = axes[3].inset_axes([0, 0, 1, 1])
    sub_ax.axis('off')
    
    # Draw simple rectangles to represent the stack
    rect_h, rect_w = 0.8, 0.6
    offsets = [0.0, 0.05, 0.1, 0.15, 0.2]
    colors = ['red', 'green', 'blue', 'cyan', 'magenta']
    labels = ['R', 'G', 'B', 'Flow$_x$', 'Flow$_y$']
    
    for i, (off, col, lbl) in enumerate(zip(offsets, colors, labels)):
        rect = plt.Rectangle((0.1+off, 0.1+off), rect_w, rect_h, facecolor=col, edgecolor='black', alpha=0.5, label=lbl)
        sub_ax.add_patch(rect)
        sub_ax.text(0.1+off+rect_w/2, 0.1+off+rect_h/2, lbl, ha='center', va='center', fontweight='bold', color='white')

    sub_ax.set_xlim(0, 1)
    sub_ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()

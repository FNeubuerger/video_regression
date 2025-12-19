import torch
import cv2
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_registry import MODEL_REGISTRY
from torchvision import transforms

def preprocess_frame(frame, prev_frame=None, target_size=(64, 64)):
    """
    Preprocess a single frame:
    1. Resize
    2. Compute Optical Flow (if prev_frame is provided)
    3. Normalize
    4. Concatenate RGB + Flow
    """
    # Resize
    frame_resized = cv2.resize(frame, target_size)
    
    # Convert to float32 and normalize RGB
    # Mean and Std from ImageNet
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    frame_norm = frame_resized.astype(np.float32) / 255.0
    frame_norm = (frame_norm - mean) / std
    
    # Transpose to (C, H, W) -> (3, 64, 64)
    frame_chw = frame_norm.transpose(2, 0, 1)
    
    # Compute Optical Flow
    if prev_frame is not None:
        prev_resized = cv2.resize(prev_frame, target_size)
        prev_gray = cv2.cvtColor(prev_resized, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        # Flow is (H, W, 2)
        # Transpose to (2, H, W)
        flow_chw = flow.transpose(2, 0, 1)
    else:
        # Zero flow for first frame
        flow_chw = np.zeros((2, target_size[0], target_size[1]), dtype=np.float32)
        
    # Concatenate RGB and Flow -> (5, 64, 64)
    input_tensor = np.concatenate([frame_chw, flow_chw], axis=0)
    
    return input_tensor

def parse_filename(filename):
    """Extract temperature from filename like frame_001_label_37.5.png"""
    import re
    match = re.search(r'label_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def create_plot(real_temps, est_temps, window_size=50):
    """Create a matplotlib plot of temperature over time."""
    fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
    
    # Get recent history
    start_idx = max(0, len(real_temps) - window_size)
    x = range(start_idx, len(real_temps))
    y_real = real_temps[start_idx:]
    y_est = est_temps[start_idx:]
    
    ax.plot(x, y_real, 'g-', label='Real', linewidth=2)
    ax.plot(x, y_est, 'r--', label='Est', linewidth=2)
    
    ax.set_ylim(30, 50) # Assuming bioheat range
    ax.set_title("Temperature History")
    ax.legend(loc='upper left', fontsize='small')
    ax.grid(True, alpha=0.3)
    
    # Convert to image
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    plt.close(fig)
    
    return image

def run_clinical_demo(model_name, checkpoint_path, sequence_dir, output_path="demo_output.mp4"):
    print(f"Loading model {model_name} from {checkpoint_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    model = ModelClass(**kwargs)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load Images
    print(f"Loading sequence from {sequence_dir}...")
    image_files = sorted([f for f in os.listdir(sequence_dir) if f.endswith('.png')])
    if not image_files:
        raise ValueError("No images found in sequence directory")
        
    # Setup Video Writer
    first_frame = cv2.imread(os.path.join(sequence_dir, image_files[0]))
    height, width = first_frame.shape[:2]
    
    # Output layout: Original Frame + Plot below or side-by-side?
    # Let's do side-by-side. 
    # Frame (W, H) + Plot (400, 200) -> We'll resize plot to match height or width.
    # Actually, let's just overlay the plot on the frame if it's large enough, or extend the canvas.
    # Let's extend the canvas to the right.
    
    plot_width = 400
    canvas_width = width + plot_width
    canvas_height = max(height, 300) # Ensure enough height for plot
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 10.0, (canvas_width, canvas_height))
    
    frame_buffer = []
    prev_frame = None
    sequence_length = 5
    
    real_temps = []
    est_temps = []
    
    print("Processing frames...")
    for i, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(sequence_dir, img_file)
        frame = cv2.imread(img_path) # BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get Ground Truth
        real_temp = parse_filename(img_file)
        if real_temp is None:
            real_temp = 0.0 # Fallback
            
        # Preprocess
        processed_frame = preprocess_frame(frame_rgb, prev_frame)
        prev_frame = frame_rgb
        
        # Update Buffer
        frame_buffer.append(processed_frame)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)
            
        # Inference
        est_temp = 0.0
        if len(frame_buffer) == sequence_length:
            input_seq = np.stack(frame_buffer) # (T, C, H, W)
            input_tensor = torch.from_numpy(input_seq).unsqueeze(0).to(device) # (1, T, C, H, W)
            
            with torch.no_grad():
                output = model(input_tensor)
                # Handle tuple output (physics models)
                if isinstance(output, tuple):
                    output = output[0]
                # Handle sequence output
                if output.dim() > 1 and output.shape[1] > 1:
                    est_temp = output[0, -1].item()
                else:
                    est_temp = output.item()
        else:
            # Not enough frames yet, use 0 or previous
            est_temp = est_temps[-1] if est_temps else 37.0
            
        real_temps.append(real_temp)
        est_temps.append(est_temp)
        
        # Visualization
        # Create Canvas
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Draw Frame
        # Resize frame to fit left side if needed, or center it
        # For now, just place at 0,0
        canvas[:height, :width] = frame
        
        # Draw Info Text
        text_x = width + 20
        cv2.putText(canvas, f"Frame: {i}", (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, f"Real: {real_temp:.1f} C", (text_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"Est:  {est_temp:.1f} C", (text_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        error = abs(real_temp - est_temp)
        color = (0, 255, 0) if error < 1.0 else (0, 165, 255) if error < 3.0 else (0, 0, 255)
        cv2.putText(canvas, f"Error: {error:.1f} C", (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw Plot
        if i > 0:
            plot_img = create_plot(real_temps, est_temps)
            # Resize plot to fit right side bottom
            plot_h, plot_w = plot_img.shape[:2]
            target_w = plot_width - 20
            scale = target_w / plot_w
            target_h = int(plot_h * scale)
            plot_resized = cv2.resize(plot_img, (target_w, target_h))
            
            # Place plot
            y_offset = 200
            if y_offset + target_h < canvas_height:
                canvas[y_offset:y_offset+target_h, text_x:text_x+target_w] = plot_resized
        
        out.write(canvas)
        
    out.release()
    print(f"Demo video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="CNNLSTM", help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--sequence", type=str, default="data/sequence_1", help="Path to sequence directory")
    parser.add_argument("--output", type=str, default="demo_output.mp4", help="Output video path")
    
    args = parser.parse_args()
    
    run_clinical_demo(args.model, args.checkpoint, args.sequence, args.output)

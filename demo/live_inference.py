import cv2
import torch
import numpy as np
import argparse
import os
import sys
import time
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Try importing onnxruntime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from utils.model_registry import MODEL_REGISTRY

def load_model(model_name, checkpoint_path, device):
    # Check for ONNX
    if checkpoint_path.endswith('.onnx'):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is not installed. Please install it to use .onnx models.")
        
        print(f"Loading ONNX model from {checkpoint_path}")
        
        # Get available providers
        available_providers = ort.get_available_providers()
        
        # Select providers based on device and availability
        providers = []
        if device.type == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            providers.append('CUDAExecutionProvider')
        if 'CPUExecutionProvider' in available_providers:
            providers.append('CPUExecutionProvider')
            
        if not providers:
            # Fallback if CPU provider is somehow missing or named differently
            providers = available_providers

        try:
            session = ort.InferenceSession(checkpoint_path, providers=providers)
        except Exception as e:
            print(f"Warning: Failed to create InferenceSession with providers {providers}. Falling back to CPU.")
            session = ort.InferenceSession(checkpoint_path, providers=['CPUExecutionProvider'])
        return session

    # PyTorch Loading
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    model = ModelClass(**kwargs)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

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
        
    # Concatenate RGB + Flow -> (5, 64, 64)
    input_tensor = np.concatenate([frame_chw, flow_chw], axis=0)
    return input_tensor

def run_live_demo(video_path, model_name, checkpoint_path, output_path=None, ground_truth_csv=None, quick_test=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = load_model(model_name, checkpoint_path, device)
    is_onnx = isinstance(model, ort.InferenceSession) if ONNX_AVAILABLE else False
    
    # Open Video or Image Sequence
    is_directory = os.path.isdir(video_path)
    image_files = []
    
    if is_directory:
        import re
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        
        def extract_frame_num(filename):
            # Try to extract number after 'frame_'
            match = re.search(r'frame_(\d+)', os.path.basename(filename))
            if match:
                return int(match.group(1))
            return 0 # Fallback

        image_files = sorted([
            os.path.join(video_path, f) for f in os.listdir(video_path) 
            if f.lower().endswith(valid_extensions)
        ], key=extract_frame_num)
        
        if not image_files:
            print(f"Error: No image files found in {video_path}")
            return
        
        # Read first frame to get properties
        first_frame = cv2.imread(image_files[0])
        if first_frame is None:
            print(f"Error: Could not read first image {image_files[0]}")
            return
            
        height, width = first_frame.shape[:2]
        video_fps = 30.0 # Default for image sequence
        total_frames = len(image_files)
        print(f"Processing {total_frames} images from directory.")
        
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
            
        # Video Properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output Video Writer
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    # Load Ground Truth if available
    gt_data = None
    gt_temps = []
    if ground_truth_csv:
        import pandas as pd
        gt_data = pd.read_csv(ground_truth_csv)
        # Assuming CSV has 'frame' and 'temperature' columns
        # Or just a list of temperatures matching frames
        if 'temperature' in gt_data.columns:
            gt_temps = gt_data['temperature'].values
        else:
            # Assume single column is temperature
            gt_temps = gt_data.iloc[:, 0].values
    elif is_directory:
        # Try to extract labels from filenames if no CSV provided
        # Format: frame_XXXX_label_YY.Y.png
        try:
            extracted_temps = []
            for img_path in image_files:
                basename = os.path.basename(img_path)
                if 'label_' in basename:
                    # Extract number after 'label_' and before extension
                    parts = basename.split('label_')
                    if len(parts) > 1:
                        temp_str = parts[1].rsplit('.', 1)[0] # Remove extension
                        extracted_temps.append(float(temp_str))
                    else:
                        extracted_temps.append(None)
                else:
                    extracted_temps.append(None)
            
            # Only use if we found labels for most frames
            if sum(1 for t in extracted_temps if t is not None) > len(image_files) * 0.5:
                gt_temps = extracted_temps
                gt_data = True # Flag to enable GT display
                print("Extracted ground truth labels from filenames.")
        except Exception as e:
            print(f"Could not extract labels from filenames: {e}")

    # Buffer for sequence models
    sequence_length = 5
    frame_buffer = []
    prev_frame = None
    
    # Inference Loop
    frame_idx = 0
    fps_history = []
    
    pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")
    
    while True:
        loop_start = time.time()
        
        # Quick Test Check
        if quick_test and frame_idx >= 150:
            print("Quick test limit reached (150 frames). Stopping.")
            break

        if is_directory:
            if frame_idx >= len(image_files):
                break
            frame = cv2.imread(image_files[frame_idx])
            if frame is None:
                print(f"Warning: Could not read image {image_files[frame_idx]}")
                frame_idx += 1
                continue
            ret = True
        else:
            if not cap.isOpened():
                break
            ret, frame = cap.read()
            if not ret:
                break
            
        # Preprocess
        processed_frame = preprocess_frame(frame, prev_frame)
        prev_frame = frame.copy()
        
        frame_buffer.append(processed_frame)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)
            
        # Run Inference
        prediction = 0.0
        inference_time = 0.0
        
        if len(frame_buffer) == sequence_length:
            inf_start = time.time()
            
            # Prepare Input: (Time, Channels, Height, Width)
            input_seq = np.stack(frame_buffer)
            # Add Batch Dim: (1, Time, Channels, Height, Width)
            input_seq = np.expand_dims(input_seq, axis=0)
            
            if is_onnx:
                # ONNX Inference
                input_name = model.get_inputs()[0].name
                input_shape = model.get_inputs()[0].shape
                
                # Check if model expects 5D (Sequence) or 4D (Single Frame)
                # ONNX shapes can have dynamic axes (strings or None), so be careful
                expects_sequence = len(input_shape) == 5
                
                if expects_sequence:
                    input_data = input_seq.astype(np.float32)
                else:
                    # If model expects 4D, take last frame: (1, C, H, W)
                    # input_seq is (1, T, C, H, W) -> (1, C, H, W)
                    input_data = input_seq[:, -1, :, :, :].astype(np.float32)
                
                outputs = model.run(None, {input_name: input_data})
                output = outputs[0] # Assume first output is prediction
                
                # Handle sequence output if any
                if output.ndim > 1 and output.shape[1] > 1:
                    prediction = output[0, -1]
                else:
                    prediction = output.item()
                    
            else:
                # PyTorch Inference
                input_tensor = torch.from_numpy(input_seq).float().to(device)
                
                # Handle Spatial Models (Single Frame) vs Temporal Models (Sequence)
                is_spatial = "ResNet" in model_name and "CNNLSTM" not in model_name
                
                with torch.no_grad():
                    output = model(input_tensor)
                    
                    # Handle tuple outputs (physics models)
                    if isinstance(output, tuple):
                        output = output[0]
                        
                    # Handle sequence output
                    if output.dim() > 1 and output.shape[1] > 1:
                        prediction = output[0, -1].item()
                    else:
                        prediction = output.item()
                        
            inference_time = (time.time() - inf_start) * 1000 # ms
            
        # Calculate FPS
        loop_time = time.time() - loop_start
        current_fps = 1.0 / loop_time if loop_time > 0 else 0
        fps_history.append(current_fps)
        if len(fps_history) > 30: fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        
        # Visualization
        # 1. Temperature Bar/Text
        text_color = (0, 255, 0) # Green
        
        # Frame Number
        cv2.putText(frame, f"Frame: {frame_idx}", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Pred: {prediction:.2f} C", (30, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        if gt_data is not None and frame_idx < len(gt_temps):
            gt_temp = gt_temps[frame_idx]
            cv2.putText(frame, f"Real: {gt_temp:.2f} C", (30, 160), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Error
            error = abs(prediction - gt_temp)
            cv2.putText(frame, f"Error: {error:.2f} C", (30, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
        # Stats Overlay
        cv2.putText(frame, f"Inf: {inference_time:.1f} ms", (30, height - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (30, height - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if output_path:
            out.write(frame)
        
        frame_idx += 1
        pbar.update(1)
            
    if not is_directory:
        cap.release()
    if output_path:
        out.release()
    pbar.close()
    cv2.destroyAllWindows()
    print(f"Demo complete. Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="CNNLSTM", help="Model name (ignored if checkpoint is .onnx)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth or .onnx)")
    parser.add_argument("--output", type=str, default="demo_output.mp4", help="Path to output video")
    parser.add_argument("--ground_truth", type=str, help="Path to ground truth CSV (optional)")
    parser.add_argument("--quick_test", action="store_true", help="Run only first 150 frames (5 seconds)")
    
    args = parser.parse_args()
    
    run_live_demo(args.video, args.model, args.checkpoint, args.output, args.ground_truth, args.quick_test)

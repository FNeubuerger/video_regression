import onnxruntime as ort
import numpy as np
import cv2
import time
import os
import argparse
import sys

# Add parent directory to path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def load_sequence(data_dir):
    """Load a sample sequence from the data directory."""
    # Find a sequence folder
    seq_dirs = [d for d in os.listdir(data_dir) if d.startswith('sequence_')]
    if not seq_dirs:
        raise ValueError("No sequence folders found in data directory")
    
    seq_path = os.path.join(data_dir, seq_dirs[0])
    images = sorted([f for f in os.listdir(seq_path) if f.endswith('.png')])[:200]
    
    frames = []
    for img_name in images:
        img_path = os.path.join(seq_path, img_name)
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    return frames

def run_demo(model_path, data_dir="data", sequence_length=3):
    print(f"Loading model from {model_path}...")
    try:
        # Use CPU provider for "limited resources" simulation
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading sample data...")
    try:
        all_frames = load_sequence(data_dir)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print(f"Loaded {len(all_frames)} frames. Starting simulation...")
    print("-" * 60)
    print(f"{'Frame':<10} | {'Inference (ms)':<15} | {'FPS':<10} | {'Prediction':<10}")
    print("-" * 60)
    
    # Buffer for sequence
    frame_buffer = []
    prev_frame = None
    
    latencies = []
    
    # Loop through frames (simulate stream)
    # We loop the sequence multiple times to get a good measurement
    for i in range(100): 
        frame_idx = i % len(all_frames)
        frame = all_frames[frame_idx]
        
        start_time = time.time()
        
        # Preprocess
        processed_frame = preprocess_frame(frame, prev_frame)
        prev_frame = frame
        
        # Add to buffer
        frame_buffer.append(processed_frame)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)
            
        # Run inference if buffer is full
        if len(frame_buffer) == sequence_length:
            # Stack frames -> (Time, Channels, Height, Width)
            input_seq = np.stack(frame_buffer)
            # Add batch dimension -> (1, Time, Channels, Height, Width)
            input_tensor = input_seq[np.newaxis, ...].astype(np.float32)
            
            # Run inference
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: input_tensor})
            # print(f"Output shape: {output[0].shape}")
            prediction = output[0].item()
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            fps = 1000 / latency_ms
            
            print(f"{i:<10} | {latency_ms:<15.2f} | {fps:<10.2f} | {prediction:.4f}")
            
        # Simulate real-time delay (optional, but we want to measure max speed)
        # time.sleep(0.03) 

    print("-" * 60)
    avg_latency = np.mean(latencies)
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average FPS: {1000/avg_latency:.2f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="demo/cnnlstm.onnx", help="Path to ONNX model")
    args = parser.parse_args()
    
    run_demo(args.model)

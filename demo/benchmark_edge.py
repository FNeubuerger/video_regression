import torch
import torch.nn as nn
import time
import numpy as np
import cv2
import os
import sys
import pandas as pd
from thop import profile
from tabulate import tabulate
import onnxruntime as ort

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from physics import PhysicsCNNLSTM
from torchvision.models import resnet18

def create_model(model_type, frame_shape, time_steps):
    """Create and return the specified model."""
    if model_type == "cnnlstm":
        return CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "pretrained_cnnlstm":
        pretrained_cnn = resnet18(weights=None) # No weights needed for FLOPs
        pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)
        return PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "simple_resnet":
        return SimpleResNet(frame_shape=frame_shape)
    elif model_type == "physics_cnnlstm":
        return PhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def measure_optical_flow_latency(height=64, width=64, iterations=100):
    """Measure the time it takes to compute optical flow on CPU."""
    prev = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    curr = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, 
            flags=0
        )
        times.append((time.time() - start) * 1000) # ms
    
    return np.mean(times), np.std(times)

def measure_inference_latency_onnx(onnx_path, input_shape, iterations=100, threads=1):
    """Measure ONNX inference latency with specific thread count."""
    if not os.path.exists(onnx_path):
        return None, None

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = threads
    sess_options.inter_op_num_threads = threads
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    try:
        session = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Failed to load {onnx_path}: {e}")
        return None, None
        
    input_name = session.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
        
    times = []
    for _ in range(iterations):
        start = time.time()
        session.run(None, {input_name: dummy_input})
        times.append((time.time() - start) * 1000)
        
    return np.mean(times), np.std(times)

def estimate_fps(flops, device_gflops):
    """
    Estimate FPS based on FLOPs and device capability.
    Note: This is a theoretical upper bound.
    """
    # Convert GFLOPS to FLOPS
    device_flops = device_gflops * 1e9
    return device_flops / flops

def main():
    frame_shape = (64, 64, 5) # 3 RGB + 2 Flow
    time_steps = 3
    
    models_config = [
        {"name": "CNNLSTM", "type": "cnnlstm", "onnx": "demo/cnnlstm.onnx", "input_shape": (1, 3, 5, 64, 64)},
        {"name": "Pretrained CNNLSTM", "type": "pretrained_cnnlstm", "onnx": "demo/pretrained_cnnlstm.onnx", "input_shape": (1, 3, 5, 64, 64)},
        # Simple ResNet expects (Batch, Channels, Height, Width) - 4D
        {"name": "Simple ResNet", "type": "simple_resnet", "onnx": "demo/simple_resnet.onnx", "input_shape": (1, 3, 5, 64, 64)},
        {"name": "Physics CNNLSTM", "type": "physics_cnnlstm", "onnx": "demo/physics_cnnlstm.onnx", "input_shape": (1, 3, 5, 64, 64)},
    ]
    
    results = []
    
    print("Measuring Optical Flow Latency...")
    flow_mean, flow_std = measure_optical_flow_latency()
    print(f"Optical Flow (64x64): {flow_mean:.2f} ms ± {flow_std:.2f}")
    
    # Hardware Profiles (Approximate GFLOPS for FP32/FP16 mixed)
    # Note: Real-world performance depends on memory bandwidth, thermal throttling, etc.
    hardware_profiles = {
        "Raspberry Pi 4 (CPU)": 13.5,  # ~13.5 GFLOPS
        "Jetson Nano (GPU)": 472,      # ~472 GFLOPS (FP16)
        "Jetson Orin Nano (GPU)": 20000 # ~20 TOPS (INT8) / ~10 TFLOPS (FP16) -> conservative 5 TFLOPS FP32 equivalent
    }
    
    for config in models_config:
        print(f"\nBenchmarking {config['name']}...")
        
        # 1. Calculate FLOPs and Params using PyTorch
        model = create_model(config['type'], frame_shape, time_steps)
        model.eval()
        
        # Create dummy input for THOP
        if config['type'] == 'simple_resnet':
            dummy_input = torch.randn(1, 5, 64, 64)
        else:
            dummy_input = torch.randn(1, 3, 5, 64, 64) # (B, T, C, H, W)
            
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # 2. Measure ONNX Latency (Simulated Limited CPU - 1 Thread)
        onnx_mean, onnx_std = measure_inference_latency_onnx(config['onnx'], config['input_shape'], threads=1)
        
        if onnx_mean is None:
            print(f"Skipping latency measurement for {config['name']} (ONNX not found)")
            onnx_mean = 0
        
        # Total Latency = Preprocessing (Flow) + Inference
        # Note: Flow is computed per frame. For sequence models, we might need flow for the newest frame.
        # Assuming pipeline: Capture -> Flow -> Inference
        total_latency = flow_mean + onnx_mean
        
        row = {
            "Model": config['name'],
            "Params (M)": params / 1e6,
            "GFLOPs": flops / 1e9,
            "Flow (ms)": flow_mean,
            "Inf (ms) [1-Core]": onnx_mean,
            "Total (ms)": total_latency,
            "Simulated FPS": 1000 / total_latency if total_latency > 0 else 0
        }
        
        # Estimate FPS for specific hardware (Theoretical)
        # We assume Flow is done on CPU and takes same time (optimistic for Pi, pessimistic for Jetson if using VPI)
        # Total Time = Flow_Time + (FLOPs / Device_GFLOPS)
        
        for hw_name, hw_gflops in hardware_profiles.items():
            # Theoretical Inference Time (ms) = (GFLOPs_Model / GFLOPs_Device) * 1000
            theoretical_inf_ms = (row["GFLOPs"] / hw_gflops) * 1000
            
            # For Jetson, we assume Flow can be accelerated or is negligible compared to CPU flow, 
            # but let's be conservative and add the CPU flow time we measured.
            # In reality, Jetson has hardware optical flow (VPI) which is much faster.
            if "Jetson" in hw_name:
                # Assume VPI flow is 5x faster
                est_flow_ms = flow_mean / 5 
            else:
                est_flow_ms = flow_mean
                
            est_total_ms = est_flow_ms + theoretical_inf_ms
            row[f"Est. FPS ({hw_name})"] = 1000 / est_total_ms
            
        results.append(row)

    # Create DataFrame and display
    df = pd.DataFrame(results)
    
    # Formatting
    pd.options.display.float_format = '{:.2f}'.format
    
    print("\n" + "="*80)
    print("EDGE DEVICE PERFORMANCE BENCHMARK")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f"))
    
    # Save to CSV
    df.to_csv("results/edge_benchmark_results.csv", index=False)
    print("\nResults saved to results/edge_benchmark_results.csv")

if __name__ == "__main__":
    main()

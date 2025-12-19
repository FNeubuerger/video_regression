import torch
import argparse
import time
import sys
import os
import json
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.emulator import EdgeEmulator
from simulation.profiles import DeviceType, PROFILES
from utils.model_registry import MODEL_REGISTRY

import copy

# Try importing onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

def benchmark_model(model_name, device_type, num_frames=50, use_onnx=False, mc_samples=10):
    """
    Benchmarks a single model on a single simulated device.
    """
    ModelClass, kwargs_orig = MODEL_REGISTRY[model_name]
    # Deepcopy kwargs to ensure fresh model instances (especially for PretrainedCNNLSTM which modifies the backbone)
    try:
        kwargs = copy.deepcopy(kwargs_orig)
    except Exception as e:
        # Fallback if deepcopy fails (e.g. unpicklable objects), though ResNet should be picklable
        print(f"Warning: Could not deepcopy kwargs for {model_name}: {e}")
        kwargs = kwargs_orig
    
    model = None
    
    if use_onnx:
        if not HAS_ONNX:
            print("Error: ONNX Runtime not installed.")
            return None
            
        onnx_path = f"models/onnx/{model_name}.onnx"
        if not os.path.exists(onnx_path):
            print(f"Error: ONNX model not found at {onnx_path}")
            return None
            
        try:
            # Load ONNX Model
            available_providers = ort.get_available_providers()
            providers = []
            if 'CUDAExecutionProvider' in available_providers and torch.cuda.is_available():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            model = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            print(f"Error loading ONNX model {model_name}: {e}")
            return None
    else:
        # Initialize PyTorch Model
        try:
            model = ModelClass(**kwargs)
            model.eval()
            if torch.cuda.is_available():
                model.to("cuda")
        except Exception as e:
            print(f"Error initializing {model_name}: {e}")
            return None

    # Wrap with Emulator
    emulator = EdgeEmulator(model, device_type)

    # Dummy Input
    # Most models take (B, T, C, H, W) or (B, C, H, W)
    # Our dataset provides (B, T, C, H, W)
    # Let's use a standard batch size of 1 for edge inference
    dummy_input = torch.randn(1, 5, 5, 64, 64)
    if torch.cuda.is_available() and not use_onnx:
        dummy_input = dummy_input.to("cuda")

    # Warmup
    try:
        if use_onnx:
            # ONNX Warmup
            input_name = model.get_inputs()[0].name
            input_numpy = dummy_input.cpu().numpy()
            for _ in range(5):
                model.run(None, {input_name: input_numpy})
        else:
            # PyTorch Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = emulator.model(dummy_input) 
    except Exception as e:
        print(f"Warmup failed for {model_name}: {e}")
        return None

    # Benchmark Loop
    emulator.total_inference_calls = 0
    emulator.total_simulated_time = 0.0
    
    start_time = time.time()
    timeout = 60.0 # 60 seconds timeout per model
    
    # Determine passes per frame (Monte Carlo samples for Bayesian models)
    is_bayesian = "Bayesian" in model_name
    passes_per_frame = mc_samples if is_bayesian else 1
    
    try:
        with torch.no_grad():
            for i in range(num_frames):
                if time.time() - start_time > timeout:
                    print(f"  Timeout reached ({timeout}s) after {i} frames.")
                    break
                
                # For Bayesian models, we run multiple passes per frame to estimate uncertainty
                for _ in range(passes_per_frame):
                    if time.time() - start_time > timeout:
                        print(f"  Timeout reached ({timeout}s) during frame {i}.")
                        break
                    _ = emulator(dummy_input)
                
                if time.time() - start_time > timeout:
                    break
                    
    except Exception as e:
        print(f"Benchmark failed for {model_name}: {e}")
        return None
            
    stats = emulator.get_stats()
    
    # Adjust stats for Bayesian models
    # The emulator counts every pass as a "call". 
    # But for the user, 1 "frame" = mc_samples "calls".
    if is_bayesian:
        # Latency per frame = Latency per pass * passes_per_frame
        stats['avg_latency_ms'] = stats['avg_latency_ms'] * passes_per_frame
        # FPS = 1000 / Latency per frame
        stats['simulated_fps'] = 1000.0 / stats['avg_latency_ms'] if stats['avg_latency_ms'] > 0 else 0.0
    
    # Add Energy Estimation
    # Energy (Joules) = Power (Watts) * Time (Seconds)
    # Energy per Frame = Power * Latency
    power = PROFILES[device_type].power_watts
    latency_sec = stats['avg_latency_ms'] / 1000.0
    energy_per_frame = power * latency_sec
    
    stats['model'] = model_name
    stats['format'] = 'ONNX' if use_onnx else 'PyTorch'
    stats['power_watts'] = power
    stats['energy_per_frame_joules'] = energy_per_frame
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Benchmark All Models on Simulated Edge Devices")
    parser.add_argument("--output", type=str, default="results/edge_benchmark_results.csv")
    parser.add_argument("--frames", type=int, default=20, help="Number of frames to average over")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX Runtime for inference")
    parser.add_argument("--mc-samples", type=int, default=10, help="Number of Monte Carlo samples for Bayesian models")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    
    results = []
    
    devices_to_test = [
        DeviceType.RASPBERRY_PI_4,
        DeviceType.JETSON_NANO,
        DeviceType.JETSON_ORIN_NANO,
        DeviceType.HIGH_END_GPU
    ]
    
    models_to_test = list(MODEL_REGISTRY.keys())
    
    print(f"Starting Benchmark Suite ({'ONNX' if args.onnx else 'PyTorch'})")
    print(f"Models: {len(models_to_test)}")
    print(f"Devices: {len(devices_to_test)}")
    print("-" * 60)
    
    for device_type in devices_to_test:
        print(f"\nSimulating Device: {PROFILES[device_type].name}")
        for model_name in tqdm(models_to_test, desc="Benchmarking Models"):
            stats = benchmark_model(model_name, device_type, num_frames=args.frames, use_onnx=args.onnx, mc_samples=args.mc_samples)
            if stats:
                results.append(stats)
                
    # Save Results
    df = pd.DataFrame(results)
    if not results:
        print("No results collected.")
        return

    df = df[['device', 'model', 'format', 'simulated_fps', 'avg_latency_ms', 'power_watts', 'energy_per_frame_joules']]
    df.to_csv(args.output, index=False)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(df.to_string())
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()

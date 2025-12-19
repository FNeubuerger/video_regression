import time
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from .profiles import DeviceProfile, DeviceType, PROFILES

# Try importing onnxruntime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

class EdgeEmulator:
    """
    Wraps a PyTorch model OR ONNX session to simulate the inference latency of an edge device.
    """
    def __init__(self, model_or_session: Union[nn.Module, 'ort.InferenceSession'], device_type: DeviceType):
        self.model = model_or_session
        self.profile = PROFILES[device_type]
        self.device_type = device_type
        self.is_onnx = HAS_ONNX and isinstance(self.model, ort.InferenceSession)
        
        # Statistics
        self.total_inference_calls = 0
        self.total_simulated_time = 0.0
        self.last_latency = 0.0
        self.start_time = None
        
        print(f"EdgeEmulator initialized for: {self.profile.name}")
        print(f"Backend: {'ONNX Runtime' if self.is_onnx else 'PyTorch'}")
        print(f"Slowdown Factor: {self.profile.slowdown_factor}x")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # 1. Measure actual inference time (on current hardware)
        t0 = time.perf_counter()
        
        if self.is_onnx:
            # ONNX Runtime Inference
            # args[0] is usually the input tensor
            # We need to convert to numpy
            input_tensor = args[0]
            if isinstance(input_tensor, torch.Tensor):
                input_numpy = input_tensor.cpu().numpy()
            else:
                input_numpy = input_tensor
                
            input_name = self.model.get_inputs()[0].name
            output = self.model.run(None, {input_name: input_numpy})
        else:
            # PyTorch Inference
            output = self.model(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
        t1 = time.perf_counter()
        actual_latency = t1 - t0
        
        # 2. Calculate target latency
        # Target = Actual * SlowdownFactor + Jitter
        jitter = np.random.normal(0, self.profile.latency_jitter)
        target_latency = (actual_latency * self.profile.slowdown_factor) + jitter
        target_latency = max(0.001, target_latency) # Ensure positive
        
        # 3. Inject Delay
        # If actual hardware is faster than target, sleep.

        sleep_time = target_latency - actual_latency
        
        if sleep_time > 0:
            time.sleep(sleep_time)
            simulated_latency = target_latency
        else:
            simulated_latency = actual_latency
            
        # 4. Update Stats
        self.total_inference_calls += 1
        self.total_simulated_time += simulated_latency
        self.last_latency = simulated_latency
        
        return output

    def get_stats(self):
        avg_latency = self.total_simulated_time / max(1, self.total_inference_calls)
        fps = 1.0 / avg_latency if avg_latency > 0 else 0.0
        return {
            "device": self.profile.name,
            "calls": self.total_inference_calls,
            "last_latency_ms": self.last_latency * 1000,
            "avg_latency_ms": avg_latency * 1000,
            "simulated_fps": fps
        }

    def print_status(self):
        stats = self.get_stats()
        print(f"\r[Edge Sim: {stats['device']}] "
              f"FPS: {stats['simulated_fps']:.1f} | "
              f"Latency: {stats['last_latency_ms']:.1f} ms", end="")

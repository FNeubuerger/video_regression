import torch
import argparse
import time
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from physics.models import PhysicsCNNLSTM
from simulation.emulator import EdgeEmulator
from simulation.profiles import DeviceType

def main():
    parser = argparse.ArgumentParser(description="Simulate Edge Deployment Performance")
    parser.add_argument("--device", type=str, default="rpi4", 
                        choices=["rpi4", "jetson_nano", "jetson_orin", "pc", "gpu"],
                        help="Target device to simulate")
    parser.add_argument("--model_path", type=str, default="models/physics_cnnlstm_model.pth",
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    # Map string to DeviceType
    device_map = {
        "rpi4": DeviceType.RASPBERRY_PI_4,
        "jetson_nano": DeviceType.JETSON_NANO,
        "jetson_orin": DeviceType.JETSON_ORIN_NANO,
        "pc": DeviceType.PC_CPU,
        "gpu": DeviceType.HIGH_END_GPU
    }
    target_device = device_map[args.device]

    print(f"=== Edge Simulation: {target_device.name} ===")
    print("Loading model...")
    
    # Initialize Model
    # Standard input shape is 5 channels (3 RGB + 2 Flow)
    model = PhysicsCNNLSTM(frame_shape=(64, 64, 5), time_steps=5, pretrained=True)
    
    # Load weights if available
    if os.path.exists(args.model_path):
        print(f"Loading weights from {args.model_path}")
        try:
            state_dict = torch.load(args.model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load weights ({e}). Using random initialization.")
    else:
        print("Checkpoint not found. Using random initialization.")

    model.eval()
    
    # Wrap with Emulator
    emulator = EdgeEmulator(model, target_device)
    
    # Dummy Input
    # (Batch=1, Time=5, Channels=5, H=64, W=64)
    dummy_input = torch.randn(1, 5, 5, 64, 64)
    
    if torch.cuda.is_available():
        emulator.to("cuda")
        dummy_input = dummy_input.to("cuda")
        print("Running simulation on Host GPU (with injected delays)...")
    else:
        print("Running simulation on Host CPU (with injected delays)...")

    print("\nStarting Simulation Loop (Press Ctrl+C to stop)...")
    print("-" * 50)
    
    try:
        while True:
            with torch.no_grad():
                _ = emulator(dummy_input)
            
            emulator.print_status()
            
    except KeyboardInterrupt:
        print("\n\nSimulation Stopped.")
        stats = emulator.get_stats()
        print("-" * 50)
        print(f"Final Report for {stats['device']}:")
        print(f"Total Frames: {stats['calls']}")
        print(f"Average Latency: {stats['avg_latency_ms']:.2f} ms")
        print(f"Simulated FPS: {stats['simulated_fps']:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()

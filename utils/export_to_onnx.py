import torch
import torch.nn as nn
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet, SpatialResNet
from physics.models import PhysicsCNNLSTM, SpatialPhysicsCNNLSTM
from utils.model_registry import MODEL_REGISTRY

def export_to_onnx(model_name, checkpoint_path, output_path, device='cpu'):
    print(f"Exporting {model_name} from {checkpoint_path} to {output_path}")
    
    # 1. Load Model
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
    
    # 2. Create Dummy Input
    # Shape: (Batch, Time, Channels, Height, Width)
    # Standard shape for our models is (1, 5, 5, 64, 64)
    # 5 frames, 5 channels (3 RGB + 2 Flow)
    dummy_input = torch.randn(1, 5, 5, 64, 64).to(device)
    
    # Handle Spatial Models (might expect 4D input if not wrapped)
    # But our training loop passes 5D input to all models, and they handle it internally.
    # However, for ONNX export, we want to be explicit.
    # If the model is purely spatial (SimpleResNet), it takes (B, C, H, W) or (B, T, C, H, W).
    # Let's stick to the 5D input as that's what the pipeline uses.
    
    # 3. Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Successfully exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., CNNLSTM)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, help="Path to output .onnx file")
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.checkpoint.replace(".pth", ".onnx")
        
    export_to_onnx(args.model, args.checkpoint, args.output)

import torch
import torch.nn as nn
import os
import sys
from torchvision.models import resnet18

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from physics import PhysicsCNNLSTM

def create_model(model_type, frame_shape, time_steps):
    """Create and return the specified model."""
    if model_type == "cnnlstm":
        return CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "pretrained_cnnlstm":
        pretrained_cnn = resnet18(weights='IMAGENET1K_V1')
        pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)
        return PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
    elif model_type == "simple_resnet":
        return SimpleResNet(frame_shape=frame_shape)
    elif model_type == "physics_cnnlstm":
        return PhysicsCNNLSTM(frame_shape=frame_shape, time_steps=time_steps, pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def export_model(model_name, model_type, frame_shape, time_steps, checkpoint_path, output_path):
    print(f"Exporting {model_name}...")
    
    # Create model
    model = create_model(model_type, frame_shape, time_steps)
    
    # Load weights
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print(f"Checkpoint not found: {checkpoint_path}. Exporting UNTRAINED model for demonstration.")

    model.eval()
    
    # Create dummy input
    # Shape: (Batch, Time, Channels, Height, Width)
    if model_type == "simple_resnet":
        # ResNet expects (Batch, Channels, Height, Width) or (Batch, Time, Channels, Height, Width)
        # But for export, we should match what the model expects.
        # SimpleResNet forward handles 5D input by taking the last frame.
        # However, for ONNX export, it's cleaner to export it as a 4D model if we only want single frame inference,
        # OR export as 5D if we want to keep the interface consistent.
        # Let's check the forward method of SimpleResNet.
        # It checks: if x.dim() == 5: x = x[:, -1, :, :, :]
        # So we can export with 5D input to be consistent with others.
        dummy_input = torch.randn(1, time_steps, frame_shape[2], frame_shape[0], frame_shape[1])
    else:
        dummy_input = torch.randn(1, time_steps, frame_shape[2], frame_shape[0], frame_shape[1])
    
    # Export to ONNX
    try:
        torch.onnx.export(
            model, 
            dummy_input, 
            output_path, 
            export_params=True, 
            opset_version=12, 
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output'], 
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Successfully exported to {output_path}")
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")

def main():
    frame_shape = (64, 64, 5)
    time_steps = 3
    
    models_to_export = [
        {
            "name": "CNNLSTM",
            "type": "cnnlstm",
            "checkpoint": "models/cnnlstm_model.pth",
            "output": "demo/cnnlstm.onnx"
        },
        {
            "name": "Pretrained CNNLSTM",
            "type": "pretrained_cnnlstm",
            "checkpoint": "models/pretrained_cnnlstm_model.pth",
            "output": "demo/pretrained_cnnlstm.onnx"
        },
        {
            "name": "Simple ResNet",
            "type": "simple_resnet",
            "checkpoint": "models/simple_resnet_model.pth",
            "output": "demo/simple_resnet.onnx"
        },
        {
            "name": "Physics CNNLSTM",
            "type": "physics_cnnlstm",
            "checkpoint": "models/physics_cnnlstm_model.pth",
            "output": "demo/physics_cnnlstm.onnx"
        }
    ]
    
    os.makedirs("demo", exist_ok=True)
    
    for m in models_to_export:
        export_model(
            m["name"], 
            m["type"], 
            frame_shape, 
            time_steps, 
            m["checkpoint"], 
            m["output"]
        )

if __name__ == "__main__":
    main()

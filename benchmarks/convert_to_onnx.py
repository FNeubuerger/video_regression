import torch
import os
import argparse
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_registry import MODEL_REGISTRY

import copy

def convert_to_onnx(output_dir="models/onnx"):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting models to ONNX in {output_dir}...")
    
    for model_name, (ModelClass, kwargs_orig) in tqdm(MODEL_REGISTRY.items(), desc="Converting"):
        try:
            # Deepcopy kwargs to avoid side effects on shared objects (like pretrained backbones)
            try:
                kwargs = copy.deepcopy(kwargs_orig)
            except Exception as e:
                print(f"  Warning: Could not deepcopy kwargs for {model_name}: {e}")
                kwargs = kwargs_orig

            # Initialize Model
            model = ModelClass(**kwargs)
            model.eval()
            
            # Try to load weights
            # We assume a naming convention or just use random weights if not found
            # Ideally we should look for the best checkpoint
            # Common names: {model_name_lower}.pth, {model_name_lower}_model.pth
            
            possible_paths = [
                f"models/{model_name.lower()}.pth",
                f"models/{model_name.lower()}_model.pth",
                f"models/best_{model_name.lower()}.pth"
            ]
            
            weights_loaded = False
            for p in possible_paths:
                if os.path.exists(p):
                    try:
                        state_dict = torch.load(p, map_location="cpu")
                        model.load_state_dict(state_dict)
                        print(f"  Loaded weights for {model_name} from {p}")
                        weights_loaded = True
                        break
                    except Exception as e:
                        print(f"  Error loading {p}: {e}")
            
            if not weights_loaded:
                print(f"  Warning: No weights found for {model_name}. Exporting with random initialization.")

            # Dummy Input
            # Most models take (B, T, C, H, W) or (B, C, H, W)
            # Our dataset provides (B, T, C, H, W)
            # Let's use a standard batch size of 1
            # Note: Some models (BayesianResNet) might expect (B, C, H, W) if they are frame-based
            # But our registry kwargs imply they are configured for 5 channels (RGB+Flow)
            
            # Check input shape expectation
            # If model handles sequence, input is 5D. If single frame, 4D.
            # Based on registry, all seem to be initialized with frame_shape=(64,64,5)
            # But some might be frame-based.
            
            # Let's inspect the forward method signature or just try 5D first
            dummy_input = torch.randn(1, 5, 5, 64, 64)
            
            # Special handling for models that might expect 4D input
            # But wait, BayesianResNet in models/bayesian.py handles 5D input by taking last frame.
            # So 5D input should be safe for all.
            
            output_path = os.path.join(output_dir, f"{model_name}.onnx")
            
            # Export
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
        except Exception as e:
            print(f"  Failed to convert {model_name}: {e}")

if __name__ == "__main__":
    convert_to_onnx()

from xml.parsers.expat import model
import cv2
import torch
import numpy as np
import argparse
import os
import sys
import time
from torchvision import transforms
from tqdm import tqdm
import matplotlib.cm as cm
from collections import deque

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_registry import MODEL_REGISTRY
from utils.xai_wrappers import RegressionWrapper

try:
    from captum.attr import LayerGradCam
except ImportError:
    print("Captum not installed. Please run 'pip install captum'")
    sys.exit(1)

def apply_heatmap(attr_map, frame, alpha=0.6):
    """
    Overlays attribution map on frame.
    attr_map: (H, W) float [0, 1]
    frame: (H, W, 3) uint8 BGR (opencv default)
    """
    # Colorize
    heatmap = cm.jet(attr_map)[:, :, :3] # RGBA -> RGB [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    
    # Overlay
    overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
    return overlay

def is_sequence_model(model_name):
    sequence_keywords = ['lstm', 'cnnlstm', 'sequence']
    return any(kw in model_name.lower() for kw in sequence_keywords)

def load_model_for_xai(model_name, checkpoint_path, device):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found")
        
    ModelClass, kwargs = MODEL_REGISTRY[model_name]
    
    # Smart Load
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Auto-detect variational
    is_variational = any("bottleneck.conv_mu" in k for k in state_dict.keys())
    if is_variational: kwargs['variational'] = True
         
    # Auto-detect channels
    detected_channels = 5 # Default
    if 'base_model.conv1.weight' in state_dict:
        w = state_dict['base_model.conv1.weight']
        detected_channels = w.shape[1]
    
    # Update kwargs
    if 'n_channels' in kwargs:
        kwargs['n_channels'] = detected_channels
    elif 'frame_shape' in kwargs:
        fs = list(kwargs['frame_shape'])
        if len(fs) == 3:
            fs[2] = detected_channels
            kwargs['frame_shape'] = tuple(fs)
            
    input_channels = detected_channels
    
    try:
        model = ModelClass(**kwargs)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"First load attempt failed: {e}. Trying fallback...")
        # Fallback to default 3 channels
        detected_channels = 3
        if 'n_channels' in kwargs: kwargs['n_channels'] = 3
        if 'frame_shape' in kwargs:
            fs = list(kwargs['frame_shape'])
            if len(fs) == 3:
                fs[2] = 3
                kwargs['frame_shape'] = tuple(fs)
        
        model = ModelClass(**kwargs)
        model.load_state_dict(state_dict)
        input_channels = 3
        
    model.to(device)
    model.eval()
    
    # Wrap
    wrapper = RegressionWrapper(model, target_mode='mean')
    wrapper.to(device)
    wrapper.eval()
    
    return model, wrapper, input_channels

def run_dashboard(video_path, model_name, checkpoint_path, output_path, device_name='cuda'):
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    model, wrapper, input_channels = load_model_for_xai(model_name, checkpoint_path, device)

    # testing which model has been loaded
    print("MODEL TYPE:", type(model))
    print("MODEL STRUCTURE:\n", model)
    
    is_seq_model = is_sequence_model(model_name)
    print(f"Sequence model detected: {is_seq_model}")
    
    # Setup GradCAM
    # Try different layers
    target_layer = None
    if hasattr(model, 'enc_layer4'): target_layer = model.enc_layer4
    elif hasattr(model, 'layer4'): target_layer = model.layer4
    elif hasattr(model, 'base_model') and hasattr(model.base_model, 'layer4'): target_layer = model.base_model.layer4
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'): target_layer = model.backbone.layer4
    
    if target_layer is None:
        print("Error: Could not find target layer for GradCAM.")
        return
        
    gradcam = LayerGradCam(wrapper, target_layer)
    
    # 2. Video Input
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 3. Writer
    # We will create a side-by-side view (Input | Attribution)
    # Output width is 2x input width (resized to 256 for visibility)
    display_size = (256, 256)
    out_w = display_size[0] * 3 # Input, Pred, Attr
    out_h = display_size[1]
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    
    normalize = transforms.Normalize(mean=[0.485], std=[0.229])
    
    frame_buffer = deque(maxlen=3)
    running_max = None
    attr_smooth = None
    
    pbar = tqdm(total=total_frames)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Preprocess
        # Resize to 64x64 for model
        input_frame = cv2.resize(frame, (64, 64))
        input_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        
        input_tensor = transforms.ToTensor()(input_rgb)
        input_tensor = normalize(input_tensor[:1] if input_tensor.shape[0] > 1 else input_tensor)
        
        if input_channels == 5:
            frame_tensor = torch.zeros((5, 64, 64))
            frame_tensor[:3] = input_tensor
        else:
            frame_tensor = input_tensor
        
        frame_buffer.append(frame_tensor)
        
        if is_seq_model:
            if len(frame_buffer) < 3:
                pbar.update(1)
                continue
            
            inp_seq = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
            inp_seq.requires_grad = True
        else:
            inp = frame_tensor.unsqueeze(0).to(device)
            inp.requires_grad = True
            inp_seq = inp
        
        # Inference & Explanation
        # GradCAM
        try:
            attr = gradcam.attribute(inp_seq)
            
            if is_seq_model:
                attr = attr.mean(dim=1)
            
            attr = LayerGradCam.interpolate(attr, display_size, interpolate_mode='bilinear')
            attr_np = attr.squeeze().detach().cpu().numpy()
            if attr_np.ndim > 2: attr_np = attr_np.mean(axis=0)
            
            curr_max = attr_np.max()
            if running_max is None: 
                running_max = curr_max if curr_max > 0 else 1.0
            else:
                running_max = 0.95 * running_max + 0.05 * curr_max if curr_max > 0 else running_max
            
            if running_max < 1e-5: running_max = 1.0
            attr_norm = np.clip(attr_np / running_max, 0, 1)
            
            if attr_smooth is None:
                attr_smooth = attr_norm
            else:
                attr_smooth = 0.7 * attr_smooth + 0.3 * attr_norm
            
            # Prediction
            with torch.no_grad():
                if is_seq_model:
                    if hasattr(model, 'variational') and model.variational:
                        pred, _ = model(inp_seq.detach())
                    else:
                        pred = model(inp_seq.detach())
                else:
                    if input_channels == 5:
                        inp_pred = torch.zeros((1, 5, 64, 64)).to(device)
                        inp_pred[0, :3] = frame_tensor.to(device)
                        if hasattr(model, 'variational') and model.variational:
                            pred, _ = model(inp_pred)
                        else:
                            pred = model(inp_pred)
                    else:
                        if hasattr(model, 'variational') and model.variational:
                            pred, _ = model(inp_seq.detach())
                        else:
                            pred = model(inp_seq.detach())
                    
            pred_map = pred.squeeze().cpu().numpy()
            pred_resized = cv2.resize(pred_map, display_size)
        except Exception as e:
            print(e)
            break
            
        # Visualization
        # 1. Original (Resized)
        vis_frame = cv2.resize(frame, display_size)
        
        # 2. Prediction Heatmap
        # Fixed normalization (20-60) to prevent flickering of the heatmap base
        pred_norm = np.clip((pred_resized - 20) / (60 - 20), 0, 1)
            
        pred_heatmap = (cm.inferno(pred_norm)[:, :, :3] * 255).astype(np.uint8)
        pred_heatmap = cv2.cvtColor(pred_heatmap, cv2.COLOR_RGB2BGR)
        
        # 3. Attribution Overlay
        attr_overlay = apply_heatmap(attr_smooth, vis_frame)
        
        # Combine
        combined = np.hstack([vis_frame, pred_heatmap, attr_overlay])
        
        # Add Text
        cv2.putText(combined, "Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(combined, "Pred Map", (display_size[0]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(combined, "XAI (CAM)", (display_size[0]*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        if output_path:
            out.write(combined)
            
        pbar.update(1)
        
    cap.release()
    if output_path: out.release()
    pbar.close()
    print(f"Dashboard video saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--model", type=str, default="ResNetUNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="dashboard_output.mp4")
    args = parser.parse_args()
    
    run_dashboard(args.video, args.model, args.checkpoint, args.output)
import cv2
import numpy as np
import torch

def compute_optical_flow(prev_frame, curr_frame):
    """
    Computes dense optical flow using Farneback's algorithm.
    
    Args:
        prev_frame (np.ndarray): Previous frame (H, W) or (H, W, C), uint8 or float.
        curr_frame (np.ndarray): Current frame (H, W) or (H, W, C), uint8 or float.
        
    Returns:
        np.ndarray: Optical flow field (H, W, 2) containing (dx, dy).
    """
    # Convert to grayscale if necessary
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = prev_frame
        
    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
    else:
        curr_gray = curr_frame
        
    # Ensure inputs are uint8 for OpenCV
    if prev_gray.dtype != np.uint8:
        prev_gray = (prev_gray * 255).astype(np.uint8)
    if curr_gray.dtype != np.uint8:
        curr_gray = (curr_gray * 255).astype(np.uint8)

    # Calculate Dense Optical Flow
    # Parameters: prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 
        pyr_scale=0.5, levels=3, winsize=15, 
        iterations=3, poly_n=5, poly_sigma=1.2, 
        flags=0
    )
    
    return flow

def preprocess_frame_with_flow(frame_sequence):
    """
    Takes a sequence of frames and returns a tensor with flow channels appended.
    
    Args:
        frame_sequence (torch.Tensor): (T, C, H, W) normalized tensor.
        
    Returns:
        torch.Tensor: (T, C+2, H, W) tensor where the last 2 channels are flow (dx, dy).
                      For the first frame, flow is zero.
    """
    T, C, H, W = frame_sequence.shape
    
    # Convert to numpy for OpenCV
    # Denormalize for flow calculation (approximate)
    frames_np = frame_sequence.permute(0, 2, 3, 1).cpu().numpy()
    frames_np = (frames_np - frames_np.min()) / (frames_np.max() - frames_np.min() + 1e-6)
    
    flow_sequence = []
    
    for t in range(T):
        if t == 0:
            # First frame has no flow, use zeros
            flow = np.zeros((H, W, 2), dtype=np.float32)
        else:
            prev = frames_np[t-1]
            curr = frames_np[t]
            flow = compute_optical_flow(prev, curr)
            
            # Normalize flow to be roughly in range [-1, 1] for the network
            # Assuming max displacement of ~20 pixels
            flow = flow / 20.0 
            
        flow_sequence.append(flow)
        
    # Stack flow: (T, H, W, 2) -> (T, 2, H, W)
    flow_tensor = torch.from_numpy(np.stack(flow_sequence)).permute(0, 3, 1, 2)
    
    # Concatenate with original frames: (T, C+2, H, W)
    return torch.cat([frame_sequence, flow_tensor], dim=1)

import torch
import torch.nn as nn
import numpy as np

class RegressionWrapper(nn.Module):
    """
    Wraps a dense regression model (outputting HxW map) to output a scalar target
    for attribution methods (Captum, Quantus).
    """
    def __init__(self, model, target_mode='mean', roi_coords=None):
        """
        Args:
            model (nn.Module): The dense regression model.
            target_mode (str): Aggregation mode. 
                               'mean' (Average temperature),
                               'max' (Peak temperature),
                               'roi' (Specific point).
            roi_coords (tuple): (x, y) coordinates for 'roi' mode.
        """
        super().__init__()
        self.model = model
        self.target_mode = target_mode
        self.roi_coords = roi_coords
        
    def forward(self, x):
        # Handle numpy input (Quantus often passes numpy)
        if isinstance(x, np.ndarray):
            device = next(self.model.parameters()).device
            x = torch.tensor(x, dtype=torch.float32).to(device)
        elif isinstance(x, torch.Tensor):
            device = next(self.model.parameters()).device
            if x.device != device:
               x = x.to(device)

        # x shape: (B, C, H, W) or (B, T, C, H, W)
        output = self.model(x)
        
        # Handle tuple return (Variational models return (pred, kl))
        if isinstance(output, tuple):
            output = output[0]

        # Check if output is already scalar (B) or (B, 1) or (B, C)
        # If it is spatial (B, C, H, W) where H,W > 1, we aggregate.
        # But checks for spatial output need to be careful.
        
        is_spatial = output.ndim >= 3 and output.shape[-1] > 1 and output.shape[-2] > 1
        
        if not is_spatial:
            # Already scalar/vector. Ensure (B, 1) for Quantus.
            if output.ndim == 1:
                output = output.unsqueeze(1)
            elif output.ndim == 2 and output.shape[1] > 1:
                 # If (B, C) and C>1, we might need to pick one? 
                 # But for now assume single target regression or handled by target_mode?
                 # Actually if simple regression, just return.
                 pass
            return output

        # Aggregation for spatial outputs
        if self.target_mode == 'mean':
            # Mean temperature across the map
            return output.mean(dim=(-2, -1)).view(output.shape[0], -1) 
            
        elif self.target_mode == 'max':
            vals = output.view(output.shape[0], -1)
            return vals.max(dim=1).values.unsqueeze(1)
            
        elif self.target_mode == 'roi':
            if self.roi_coords is None:
                raise ValueError("roi_coords must be provided for 'roi' mode")
            x, y = self.roi_coords
            # Check bounds
            if x < 0 or x >= output.shape[-1] or y < 0 or y >= output.shape[-2]:
                 raise ValueError(f"ROI coords {self.roi_coords} out of bounds for output {output.shape}")
            return output[..., y, x].unsqueeze(1)
            
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

class TimeDistributedWrapper(nn.Module):
    """
    Wraps a 2D CNN (frame-by-frame) to handle 5D input (B, T, C, H, W)
    by processing frames independently and stacking results.
    Useful for attributing to specific frames in a sequence for a 2D model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)
        out = self.model(x_reshaped)
        # Assuming out is (B*T, 1, H, W)
        return out.view(b, t, 1, h, w)

import torch
import torch.nn as nn

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
        # x shape: (B, C, H, W) or (B, T, C, H, W)
        output = self.model(x)
        
        # Handle tuple return (Variational models return (pred, kl))
        if isinstance(output, tuple):
            output = output[0]
        
        # Ensure output is (B, H, W) or (B, 1, H, W)
        if output.dim() == 4 and output.shape[1] == 1:
            output = output.squeeze(1)
        
        # If output is already scalar/vector (B, ) or (B, 1), just return it
        if output.dim() <= 2:
            if output.dim() == 1:
                 return output.unsqueeze(1) # (B, 1)
            return output # (B, 1)

        # Aggregation
        if self.target_mode == 'mean':
            # Mean temperature across the map
            return output.mean(dim=(1, 2)).unsqueeze(1) # Return (B, 1)
            
        elif self.target_mode == 'max':
            # Peak temperature (differentiable approx or hard max?)
            # Hard max is fine for gradients usually, but let's just take max value
            # Note: Captum expects scalar per batch item
            vals = output.view(output.shape[0], -1)
            return vals.max(dim=1).values.unsqueeze(1)
            
        elif self.target_mode == 'roi':
            if self.roi_coords is None:
                raise ValueError("roi_coords must be provided for 'roi' mode")
            x, y = self.roi_coords
            # Check bounds
            if x < 0 or x >= output.shape[2] or y < 0 or y >= output.shape[1]:
                 raise ValueError(f"ROI coords {self.roi_coords} out of bounds for output {output.shape}")
            return output[:, y, x].unsqueeze(1)
            
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

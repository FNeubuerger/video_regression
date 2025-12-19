import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss function for Temperature Estimation.
    
    Combines:
    1. MSE Loss (Data fidelity)
    2. Temporal Smoothness Loss (Physics constraint: Temperature changes smoothly)
    3. Monotonicity Loss (Optional: If we know heating is active)
    """
    def __init__(self, smoothness_weight=0.1, monotonicity_weight=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.smoothness_weight = smoothness_weight
        self.monotonicity_weight = monotonicity_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, time_steps) or (batch_size, 1)
            targets: Tensor of shape (batch_size, time_steps) or (batch_size, 1)
        """
        # 1. Data Fidelity (MSE)
        # If predictions are sequence but targets are scalar (last frame), take last prediction
        if predictions.dim() > 1 and predictions.shape[1] > 1 and targets.dim() == 1:
            data_loss = self.mse(predictions[:, -1], targets)
        elif predictions.dim() > 1 and predictions.shape[1] > 1 and targets.dim() == 2 and targets.shape[1] == 1:
             data_loss = self.mse(predictions[:, -1], targets.squeeze(-1))
        else:
            data_loss = self.mse(predictions, targets)
            
        total_loss = data_loss
        
        # 2. Temporal Smoothness (Only if we predict a sequence)
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            # Minimize the first derivative (penalize rapid changes)
            # dT/dt should be small/smooth
            diffs = predictions[:, 1:] - predictions[:, :-1]
            smoothness_loss = torch.mean(diffs ** 2)
            total_loss += self.smoothness_weight * smoothness_loss
            
            # 3. Monotonicity (Optional)
            # If we assume heating, diffs should be >= 0
            # Penalize negative diffs
            if self.monotonicity_weight > 0:
                negative_diffs = torch.relu(-diffs) # Only positive if diff is negative
                monotonicity_loss = torch.mean(negative_diffs ** 2)
                total_loss += self.monotonicity_weight * monotonicity_loss
                
        return total_loss

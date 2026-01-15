import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    Physics-Informed Loss function for Temperature Estimation.
    
    Combines:
    1. MSE Loss (Data fidelity)
    2. Newton's Law of Cooling (Physics constraint)
       dT/dt = -k * (T - T_env)
       Residual = | dT/dt + k * (T - T_env) |^2
    3. Monotonicity Loss (Optional)
    """
    def __init__(self, physics_weight=0.1, k=0.1, T_env=25.0, monotonicity_weight=0.0, smoothness_weight=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight
        self.k = k
        self.T_env = T_env
        self.monotonicity_weight = monotonicity_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, predictions, targets, mask=None):
        """
        Args:
            predictions: Tensor of shape (batch_size, time_steps)
            targets: Tensor of shape (batch_size, time_steps) or (batch_size, 1)
            mask: Optional mask (ignored for scalar predictions)
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
        
        # 2. Physics Constraint (Newton's Law of Cooling)
        # Only applicable if we predict a sequence
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            # Calculate dT/dt (finite difference)
            # Shape: (batch, time_steps-1)
            dT_dt = predictions[:, 1:] - predictions[:, :-1]
            
            # Calculate T (using the earlier time step for the state)
            # Shape: (batch, time_steps-1)
            T_current = predictions[:, :-1]
            
            # Physics Residual: dT/dt + k(T - T_env) = 0
            # We want to minimize this residual
            residual = dT_dt + self.k * (T_current - self.T_env)
            physics_loss = torch.mean(residual ** 2)
            
            total_loss += self.physics_weight * physics_loss
            
            # 3. Monotonicity (Optional)
            if self.monotonicity_weight > 0:
                # If heating, dT/dt should be positive
                # Penalty if dT/dt < 0
                mono_loss = torch.mean(torch.relu(-dT_dt))
                total_loss += self.monotonicity_weight * mono_loss
            
            # 4. Smoothness (Optional - Temporal Smoothness for scalar)
            if self.smoothness_weight > 0:
                # d2T/dt2 (finite difference)
                # (T[i+1] - 2T[i] + T[i-1]) / dt^2
                if predictions.shape[1] > 2:
                    d2T_dt2 = predictions[:, 2:] - 2*predictions[:, 1:-1] + predictions[:, :-2]
                    smooth_loss = torch.mean(d2T_dt2 ** 2)
                    total_loss += self.smoothness_weight * smooth_loss
                
        return total_loss

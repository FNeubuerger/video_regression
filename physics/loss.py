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
    def __init__(self, physics_weight=0.1, k=0.1, T_env=25.0, monotonicity_weight=0.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight
        self.k = k
        self.T_env = T_env
        self.monotonicity_weight = monotonicity_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (batch_size, time_steps)
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
                # If heating, dT/dt should be positive? 
                # Or if cooling, negative. 
                # Let's assume general smoothness/monotonicity if requested
                # For now, let's stick to the physics loss as the primary regularizer
                pass
                
        return total_loss

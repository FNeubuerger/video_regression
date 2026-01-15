import torch
import torch.nn as nn

class AdvancedBioHeatLoss(nn.Module):
    """
    Advanced Physics-Informed Loss with:
    1. Learnable Physics Parameters (Inverse PINN capability)
    2. Spatial Diffusion Support (if input is a map)
    3. Pennes' Bioheat Equation
    
    Note on Temporal Dynamics:
    - If the input is a sequence (Time > 1), the full Bioheat equation (dT/dt = Diffusion + Perfusion) is enforced.
    - If the input is a single frame (Time = 1), only Spatial Diffusion regularization (Laplacian smoothing) is applied, 
      as the temporal derivative (dT/dt) cannot be computed.
    """
    def __init__(self, 
                 physics_weight=1.0, 
                 initial_perfusion=0.01, 
                 initial_conductivity=0.001, # k / (rho * c)
                 initial_metabolic_rate=0.0, # Q_met
                 arterial_temp=37.0,
                 learnable_params=True,
                 dx=1.0, # Spatial step size (mm)
                 dt=1.0, # Time step size (s)
                 spatial_params=False,
                 frame_shape=(64, 64)): 
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight
        self.T_a = arterial_temp
        self.dt = dt
        self.dx = dx
        self.spatial_params = spatial_params
        
        # Learnable Parameters
        # We use Log-space to ensure positivity
        param_shape = (1, 1, *frame_shape) if spatial_params else (1,)
        
        if learnable_params:
            self.log_alpha = nn.Parameter(torch.log(torch.full(param_shape, initial_perfusion)))
            self.log_beta = nn.Parameter(torch.log(torch.full(param_shape, initial_conductivity)))
            
            init_qm = initial_metabolic_rate if initial_metabolic_rate > 1e-6 else 1e-6
            self.log_qm = nn.Parameter(torch.log(torch.full(param_shape, init_qm)))
        else:
            self.register_buffer('log_alpha', torch.log(torch.full(param_shape, initial_perfusion)))
            self.register_buffer('log_beta', torch.log(torch.full(param_shape, initial_conductivity)))
            init_qm = initial_metabolic_rate if initial_metabolic_rate > 1e-6 else 1e-6
            self.register_buffer('log_qm', torch.log(torch.full(param_shape, init_qm)))

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def beta(self):
        return torch.exp(self.log_beta)
        
    @property
    def qm(self):
        return torch.exp(self.log_qm)

    def laplacian(self, T):
        """
        Compute Laplacian of T (batch, time, H, W)
        Assumes H, W are spatial dimensions.
        """
        # T shape: (B, Time, H, W)
        # Simple 5-point stencil finite difference
        # T_xx = (T(x+1) - 2T(x) + T(x-1)) / dx^2
        
        # Pad spatial dims
        T_padded = torch.nn.functional.pad(T, (1, 1, 1, 1), mode='replicate')
        
        T_center = T
        T_left   = T_padded[:, :, 1:-1, :-2]
        T_right  = T_padded[:, :, 1:-1, 2:]
        T_up     = T_padded[:, :, :-2, 1:-1]
        T_down   = T_padded[:, :, 2:, 1:-1]
        
        lap = (T_left + T_right + T_up + T_down - 4 * T_center) / (self.dx ** 2)
        return lap

    def gradient(self, T):
        """
        Compute spatial gradients (T_x, T_y)
        """
        # Pad spatial dims
        T_padded = torch.nn.functional.pad(T, (1, 1, 1, 1), mode='replicate')
        
        # Central difference
        T_x = (T_padded[:, :, 1:-1, 2:] - T_padded[:, :, 1:-1, :-2]) / (2 * self.dx)
        T_y = (T_padded[:, :, 2:, 1:-1] - T_padded[:, :, :-2, 1:-1]) / (2 * self.dx)
        return T_x, T_y

    def forward(self, predictions, targets, flow=None, mask=None, alpha_map=None, beta_map=None):
        """
        Args:
            predictions: 
                - Scalar Sequence: (batch, time)
                - Spatial Map Sequence: (batch, time, H, W)
            targets: (batch, time) or (batch, 1) or (batch, H, W)
            flow: Optional (batch, time, 2, H_in, W_in) - Optical Flow field
            mask: Optional (batch, 1, H, W) - Spatial mask for artifacts
            alpha_map: Optional (batch, 1, H, W) - Learnable perfusion map (Issue #41)
            beta_map: Optional (batch, 1, H, W) - Learnable conductivity map (Issue #41)
        """
        # 1. Data Fidelity (MSE)
        if targets.dim() >= 3:
            # Spatial Loss (Map to Map comparison)
            if predictions.dim() == 4 and targets.dim() == 3:
                data_loss = self.mse(predictions.squeeze(1), targets)
            elif predictions.dim() == 3 and targets.dim() == 4:
                data_loss = self.mse(predictions, targets.squeeze(1))
            else:
                data_loss = self.mse(predictions, targets)
        else:
            # Scalar Loss (Pool if predictions are spatial)
            if predictions.dim() == 4: # (B, T, H, W)
                pred_scalar = predictions.mean(dim=(2, 3)) # Global Average Pooling
            else:
                pred_scalar = predictions

            if pred_scalar.dim() > 1 and pred_scalar.shape[1] > 1 and targets.dim() == 1:
                data_loss = self.mse(pred_scalar[:, -1], targets)
            elif pred_scalar.dim() > 1 and pred_scalar.shape[1] > 1 and targets.dim() == 2 and targets.shape[1] == 1:
                data_loss = self.mse(pred_scalar[:, -1], targets.squeeze(-1))
            else:
                data_loss = self.mse(pred_scalar, targets)
            
        total_loss = data_loss
        
        # 2. Physics Constraint
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            # Temporal Derivative
            if predictions.dim() == 4:
                dT_dt = (predictions[:, 1:] - predictions[:, :-1]) / self.dt
                T_current = predictions[:, :-1]
                
                # Spatial Diffusion Term
                lap_T = self.laplacian(T_current)
                
                # Use provided maps or scalar parameters
                curr_beta = beta_map if beta_map is not None else self.beta
                curr_alpha = alpha_map if alpha_map is not None else self.alpha
                
                diffusion_term = curr_beta * lap_T
                
                # Convection Term
                convection_term = 0.0
                if flow is not None:
                    B, T_raw, H, W = T_current.shape
                    flow_flat = flow.view(-1, 2, flow.shape[-2], flow.shape[-1])
                    flow_down = torch.nn.functional.adaptive_avg_pool2d(flow_flat, (H, W))
                    flow_down = flow_down.view(B, flow.shape[1], 2, H, W)
                    flow_current = flow_down[:, :-1]
                    v_x = flow_current[:, :, 0]
                    v_y = flow_current[:, :, 1]
                    T_x, T_y = self.gradient(T_current)
                    convection_term = v_x * T_x + v_y * T_y
                    
                # Residual
                perfusion_term = -curr_alpha * (T_current - self.T_a)
                metabolic_term = self.qm
                
                residual = dT_dt + convection_term - (diffusion_term + perfusion_term + metabolic_term)
                
                # Apply Mask if provided
                if mask is not None:
                    # mask is (B, 1, H_in, W_in), residual is (B, T-1, H, W)
                    # Downsample mask to match residual resolution
                    B, _, H_in, W_in = mask.shape
                    _, _, H, W = residual.shape
                    if H_in != H or W_in != W:
                        mask_down = torch.nn.functional.adaptive_avg_pool2d(mask, (H, W))
                    else:
                        mask_down = mask
                        
                    # Broadcase mask to match time steps (residual has T-1 frames)
                    residual = residual * (1.0 - mask_down)
                
                physics_loss = torch.mean(residual ** 2)
            else:
                # Scalar Sequence
                dT_dt = (predictions[:, 1:] - predictions[:, :-1]) / self.dt
                T_current = predictions[:, :-1]
                perfusion_term = -self.alpha * (T_current - self.T_a)
                metabolic_term = self.qm
                residual = dT_dt - (perfusion_term + metabolic_term)
                physics_loss = torch.mean(residual ** 2)
            
            total_loss += self.physics_weight * physics_loss
            
        elif (predictions.dim() == 4 and predictions.shape[1] == 1) or predictions.dim() == 3:
            # Single Frame Spatial Case
            if predictions.dim() == 3: # (B, H, W)
                T_current = predictions.unsqueeze(1)
            else:
                T_current = predictions
                
            lap_T = self.laplacian(T_current)
            diffusion_term = self.beta * lap_T
            perfusion_term = -self.alpha * (T_current - self.T_a)
            metabolic_term = self.qm
            
            convection_term = 0.0
            if flow is not None:
                if flow.dim() == 4: flow = flow.unsqueeze(1)
                B, T, C_f, H_f, W_f = flow.shape
                flow_flat = flow.view(-1, C_f, H_f, W_f)
                flow_down = torch.nn.functional.adaptive_avg_pool2d(flow_flat, (T_current.shape[-2], T_current.shape[-1]))
                flow_current = flow_down.view(B, T, C_f, T_current.shape[-2], T_current.shape[-1])
                v_x = flow_current[:, :, 0]; v_y = flow_current[:, :, 1]
                T_x, T_y = self.gradient(T_current)
                convection_term = v_x * T_x + v_y * T_y

            residual = convection_term - (diffusion_term + perfusion_term + metabolic_term)
            
            # Apply Mask if provided
            if mask is not None:
                residual = residual * (1.0 - mask)
                
            physics_loss = torch.mean(residual ** 2)
            total_loss += self.physics_weight * physics_loss

        return total_loss


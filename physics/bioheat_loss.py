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
                 dt=1.0): # Time step size (s)
        super().__init__()
        self.mse = nn.MSELoss()
        self.physics_weight = physics_weight
        self.T_a = arterial_temp
        self.dt = dt
        self.dx = dx
        
        # Learnable Parameters
        # We use Log-space to ensure positivity
        if learnable_params:
            self.log_alpha = nn.Parameter(torch.log(torch.tensor(initial_perfusion)))
            self.log_beta = nn.Parameter(torch.log(torch.tensor(initial_conductivity)))
            # Metabolic rate can be zero, so we don't use log space for it, or we use a small epsilon
            # Let's assume it's positive and use log space for stability, initialized to a small value if 0
            init_qm = initial_metabolic_rate if initial_metabolic_rate > 1e-6 else 1e-6
            self.log_qm = nn.Parameter(torch.log(torch.tensor(init_qm)))
        else:
            self.register_buffer('log_alpha', torch.log(torch.tensor(initial_perfusion)))
            self.register_buffer('log_beta', torch.log(torch.tensor(initial_conductivity)))
            init_qm = initial_metabolic_rate if initial_metabolic_rate > 1e-6 else 1e-6
            self.register_buffer('log_qm', torch.log(torch.tensor(init_qm)))

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

    def forward(self, predictions, targets, flow=None):
        """
        Args:
            predictions: 
                - Scalar Sequence: (batch, time)
                - Spatial Map Sequence: (batch, time, H, W)
            targets: (batch, time) or (batch, 1)
            flow: Optional (batch, time, 2, H_in, W_in) - Optical Flow field
        """
        # 1. Data Fidelity (MSE)
        # If predictions are spatial, we need to aggregate them to match scalar targets
        # Assumption: The scalar target corresponds to the MEAN temperature of the ROI (or center)
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
                # div(k grad T) -> beta * Laplacian(T)
                lap_T = self.laplacian(T_current)
                diffusion_term = self.beta * lap_T
                
                # Convection Term (if flow is provided)
                convection_term = 0.0
                if flow is not None:
                    # Flow is usually (B, T, 2, H_in, W_in)
                    # We need to downsample it to match T (B, T, H, W)
                    # T is usually 4x4, Flow is 64x64
                    B, T, H, W = T_current.shape
                    
                    # Reshape flow for interpolation: (B*T, 2, H_in, W_in)
                    flow_flat = flow.view(-1, 2, flow.shape[-2], flow.shape[-1])
                    flow_down = torch.nn.functional.adaptive_avg_pool2d(flow_flat, (H, W))
                    flow_down = flow_down.view(B, flow.shape[1], 2, H, W)
                    
                    # Align flow time with T_current (remove last frame if needed, or first?)
                    # Flow[t] is usually flow from t-1 to t.
                    # T_current is T[0...N-1]. dT/dt is T[1...N] - T[0...N-1].
                    # We should use flow corresponding to the interval.
                    # Let's assume flow input matches T sequence length.
                    flow_current = flow_down[:, :-1] # Match T_current time steps
                    
                    v_x = flow_current[:, :, 0]
                    v_y = flow_current[:, :, 1]
                    
                    T_x, T_y = self.gradient(T_current)
                    
                    # v . grad(T)
                    convection_term = v_x * T_x + v_y * T_y
                    
            else:
                # Scalar case (Lumped parameter)
                dT_dt = (predictions[:, 1:] - predictions[:, :-1]) / self.dt
                T_current = predictions[:, :-1]
                diffusion_term = 0.0 # No spatial info
                convection_term = 0.0
            
            # Perfusion Term: - alpha * (T - T_a)
            perfusion_term = -self.alpha * (T_current - self.T_a)
            
            # Metabolic Heat Generation Term: + qm
            metabolic_term = self.qm
            
            # Residual
            # dT/dt + v.grad(T) = Diffusion + Perfusion + Metabolic + Source
            # Residual = dT/dt + Convection - (Diffusion + Perfusion + Metabolic)
            residual = dT_dt + convection_term - (diffusion_term + perfusion_term + metabolic_term)
            
            physics_loss = torch.mean(residual ** 2)
            
            total_loss += self.physics_weight * physics_loss
            
        elif (predictions.dim() == 4 and predictions.shape[1] == 1) or predictions.dim() == 3:
            # Single Frame Spatial Case (e.g. SpatialResNet on single image)
            # We cannot compute dT/dt, so we cannot enforce the full Bioheat equation.
            # However, we can enforce the Steady-State Bioheat Equation:
            # 0 = Diffusion + Perfusion + Metabolic - Convection
            # Or just Spatial Smoothness if we want to be simple.
            
            if predictions.dim() == 3: # (B, H, W)
                T_current = predictions.unsqueeze(1)
            else:
                T_current = predictions
                
            # Spatial Diffusion Term
            lap_T = self.laplacian(T_current)
            diffusion_term = self.beta * lap_T
            
            # Perfusion Term
            perfusion_term = -self.alpha * (T_current - self.T_a)
            
            # Metabolic Term
            metabolic_term = self.qm
            
            # Convection Term (if flow is provided)
            convection_term = 0.0
            if flow is not None:
                # Flow is usually (B, 1, 2, H_in, W_in) or (B, 2, H_in, W_in)
                if flow.dim() == 5:
                    flow_current = flow
                elif flow.dim() == 4:
                    flow_current = flow.unsqueeze(1)
                
                # Downsample if needed
                if flow_current.shape[-1] != T_current.shape[-1]:
                     B_f, T_f, C_f, H_f, W_f = flow_current.shape
                     flow_flat = flow_current.view(-1, C_f, H_f, W_f)
                     flow_down = torch.nn.functional.adaptive_avg_pool2d(flow_flat, (T_current.shape[-2], T_current.shape[-1]))
                     flow_current = flow_down.view(B_f, T_f, C_f, T_current.shape[-2], T_current.shape[-1])

                v_x = flow_current[:, :, 0]
                v_y = flow_current[:, :, 1]
                T_x, T_y = self.gradient(T_current)
                convection_term = v_x * T_x + v_y * T_y

            # Steady State Residual:
            # 0 = Diffusion + Perfusion + Metabolic - Convection
            # Residual = Convection - (Diffusion + Perfusion + Metabolic)
            residual = convection_term - (diffusion_term + perfusion_term + metabolic_term)
            
            physics_loss = torch.mean(residual ** 2)
            
            total_loss += self.physics_weight * physics_loss

        return total_loss, self.alpha.item(), self.beta.item()


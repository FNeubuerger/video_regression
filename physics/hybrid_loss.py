import torch
import torch.nn as nn
import torch.nn.functional as F

class BioheatHybridLoss(nn.Module):
    """
    Hybrid Loss = Data_Loss + lambda * Physics_Loss
    
    Physics Component enforces the Bioheat Transfer Equation (Steady State or Spatial Smoothness).
    
    Residual R = k * Laplacian(T) - w_b * c_b * (T - T_arterial) [+ Q_metabolic + Q_source]
    
    Since we don't know Q_source exact distribution in the loss (unless passed),
    we might calculate this loss PRIMARILY in regions away from the center, 
    or assume the network handles the source via the data term.
    """
    def __init__(self, 
                 lambda_physics=1e-4, 
                 dx=0.0006, # ~0.6 mm per pixel
                 k=0.5,     # Thermal conductivity (W/m/K)
                 w_b=0.005, # Perfusion rate (1/s) - rough approx for high perfusion
                 c_b=3600,  # Blood heat capacity (J/kg/K)
                 density=1000, 
                 t_arterial=37.0,
                 device='cuda'):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.dx = dx
        self.k = k
        # Perfusion coefficient: beta = w_b * rho_b * c_b
        # Approx: 0.005 * 1000 * 3600 ~ 18000? That seems high.
        # Check standard values: w_b ~ 0.5-5 kg/m3/s. 
        # Let's use a learnable or smaller heuristic parameter for now.
        self.beta = 1000.0 
        self.t_art = t_arterial
        
        # Laplacian Kernel
        self.laplacian_kernel = torch.tensor([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    def forward(self, pred_map, target_sparse, mask_sparse):
        """
        pred_map: (B, 1, H, W) Dense Prediction
        target_sparse: (B, 1, H, W) Sparse Ground Truth
        mask_sparse: (B, 1, H, W) 1 where GT exists
        """
        # 1. Data Loss (Sparse MSE)
        diff = (pred_map - target_sparse) * mask_sparse
        mse_loss = (diff ** 2).sum() / mask_sparse.sum().clamp(min=1.0)
        
        if self.lambda_physics == 0:
            return mse_loss, mse_loss.item(), 0.0

        # 2. Physics Loss (PDE Residual)
        # We compute Laplacian of Prediction
        # Boundary conditions: replicate padding to avoid edge artifacts
        pred_padded = F.pad(pred_map, (1, 1, 1, 1), mode='replicate')
        laplacian = F.conv2d(pred_padded, self.laplacian_kernel) # (B, 1, H, W)
        
        # Scale Laplacian by dx^2
        laplacian_phys = laplacian / (self.dx ** 2)
        
        # Conduction Term
        conduction = self.k * laplacian_phys
        
        # Perfusion Term (Decay)
        # - beta * (T - Ta)
        perfusion = -self.beta * (pred_map - self.t_art)
        
        # Residual = Conduction + Perfusion (should be 0 in non-source regions)
        residual = conduction + perfusion
        
        physics_loss = torch.mean(residual ** 2)
        
        total_loss = mse_loss + self.lambda_physics * physics_loss
        
        return total_loss, mse_loss.item(), physics_loss.item()
        return total_loss, mse_loss.item(), physics_loss.item()

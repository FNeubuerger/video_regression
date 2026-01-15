import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLTCCell(nn.Module):
    """
    Convolutional Liquid Time Constant (LTC) Cell.
    Maintains spatial structure while evolving temporally using ODE dynamics.
    
    ODE: dH/dt = -(1/tau + sigmoid(Conv(x,h))) * H + tanh(Conv(x,h))
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, ode_solver='euler'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.ode_solver = ode_solver
        padding = kernel_size // 2
        
        # We need to compute:
        # 1. Input-dependent decay factor (gamma)
        # 2. Input-dependent drive signal (mu)
        
        # Combined Conv for efficiency: Input -> [Gamma, Mu]
        # We concatenate x and h
        self.conv_gates = nn.Conv2d(
            in_channels + hidden_channels, 
            2 * hidden_channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # Base leak (1/tau) - Learnable parameter per channel
        self.base_leak = nn.Parameter(torch.ones(1, hidden_channels, 1, 1) * 0.5)
        
    def forward(self, x, h_prev, dt=1.0):
        """
        x: (B, in_channels, H, W)
        h_prev: (B, hidden_channels, H, W)
        dt: float or tensor, time step size
        """
        # Concatenate along channel dim
        combined = torch.cat([x, h_prev], dim=1)
        
        # Compute gates
        gates = self.conv_gates(combined)
        gamma, mu = torch.split(gates, self.hidden_channels, dim=1)
        
        # Gamma: Input-dependent leak. Constrain to > 0.
        # We use Sigmoid to bound it, ensuring stability.
        # Total Leak = Base_Leak + Sigmoid(Gamma)
        total_leak = torch.abs(self.base_leak) + torch.sigmoid(gamma)
        
        # Mu: Drive signal. Tanh for bounded range [-1, 1]
        drive = torch.tanh(mu)
        
        # ODE Update: dH = (-Leak * H + Drive) * dt
        if self.ode_solver == 'euler':
            dh = (-total_leak * h_prev + drive) * dt
            h_new = h_prev + dh
        else:
            # Placeholder for RK4 or simpler semi-implicit
            dh = (-total_leak * h_prev + drive) * dt
            h_new = h_prev + dh
            
        return h_new

class ConvLTC(nn.Module):
    """
    Sequence-level wrapper for ConvLTCCell.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLTCCell(in_channels, hidden_channels, kernel_size)
        
    def forward(self, x, h_init=None):
        """
        x: (B, T, C, H, W)
        Returns: (B, T, Hidden, H, W)
        """
        b, t, c, h, w = x.size()
        
        if h_init is None:
            state = torch.zeros(b, self.hidden_channels, h, w, device=x.device)
        else:
            state = h_init
            
        outputs = []
        
        for step in range(t):
            xt = x[:, step]
            state = self.cell(xt, state)
            outputs.append(state)
            
        # Stack temporal dim
        return torch.stack(outputs, dim=1), state

class ConvLTC_Model(nn.Module):
    """
    End-to-End Model:
    Encoder (CNN) -> ConvLTC -> Decoder (1x1 Conv)
    """
    def __init__(self, input_channels=3, hidden_channels=32, output_channels=1):
        super().__init__()
        
        # 1. Feature Extractor (Encoder)
        # Keeps spatial resolution or downsamples slightly?
        # Let's keep resolution for now (High Resolution processing)
        # or downsample 2x to save memory then upsample.
        # Let's try 1x1 stride (no downsampling) for pure physics fidelity first.
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 2. Dynamics
        self.core = ConvLTC(in_channels=32, hidden_channels=hidden_channels)
        
        # 3. Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, output_channels, kernel_size=1) # pointwise mapping
        )
        
    def forward(self, x):
        """
        x: (B, T, 3, H, W)
        """
        b, t, c, h, w = x.size()
        
        # Flatten time
        x_flat = x.view(b * t, c, h, w)
        
        # Encode
        feats = self.encoder(x_flat)
        _, cf, hf, wf = feats.size()
        
        # Reshape for RNN
        feats_seq = feats.view(b, t, cf, hf, wf)
        
        # ConvLTC
        core_out, _ = self.core(feats_seq) # (B, T, Hidden, H, W)
        
        # Decode
        core_flat = core_out.view(b * t, -1, hf, wf)
        out_flat = self.decoder(core_flat) # (B*T, 1, H, W)
        
        # Reshape Output to (B, T, 1, H, W)
        # Note: The model currently outputs a Spatial Map (B, T, 1, H, W)
        # But training/evaluation might expect (B, T, 4) if using scalar targets
        out = out_flat.view(b, t, -1, hf, wf)
        
        # Scalar regression adapter (average over sensor locations or global average?)
        # For now, let's keep spatial output for Physics Loss support.
        # But if we want 4 scalars, we need an adapter.
        # Since LTC U-Net is described as a 'Dense' approach, it likely stays spatial.
        
        return out

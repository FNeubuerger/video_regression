import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from torchvision.models import resnet18

class VariationalEncoder(nn.Module):
    """Encodes Features to Gaussian Distribution N(mu, sigma)"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)
        
    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        if self.training:
            z = mu + std * eps
        else:
            z = mu + std * eps
            
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        
        return z, kl_loss

class LatentLTC_UNet(nn.Module):
    """
    Architecture B from Research Part 2: Latent-Space Dynamics.
    
    1. Encoder (ResNet): Compresses frame to latent vector z.
    2. Dynamics (LTC): Evolves z_t -> z_{t+1} using Physics-Inspired ODEs.
    3. Decoder (UpConv): Reconstructs dense heatmap from z_{t+1}.
    """
    def __init__(self, n_channels=3, latent_dim=128, ncp_units=32, variational=False):
        super(LatentLTC_UNet, self).__init__()
        
        self.latent_dim = latent_dim
        self.variational = variational
        
        # Validation for AutoNCP
        # AutoNCP requires units > output_size (latent_dim)
        # We enforce this or adjust
        if ncp_units <= latent_dim:
            print(f"Warning: ncp_units ({ncp_units}) <= latent_dim ({latent_dim}). Increasing units to {latent_dim + 32}")
            ncp_units = latent_dim + 32
            
        # --- 1. Encoder ---
        # Use ResNet18 until the average pooling layer
        resnet = resnet18(weights='IMAGENET1K_V1')
        
        # Modify input channels if needed
        if n_channels != 3:
            resnet.conv1 = nn.Conv2d(
                n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
        # Remove fc layer, keep everything else
        self.encoder_backbone = nn.Sequential(*list(resnet.children())[:-1]) # Output: (B, 512, 1, 1)
        
        # Project to latent space
        if self.variational:
            self.vae_encoder = VariationalEncoder(512, latent_dim)
        else:
            self.fc_encode = nn.Linear(512, latent_dim)
        
        # --- 2. LTC Dynamics ---
        # Neural Circuit Policy wiring
        # AutoNCP(units, output_neurons)
        wiring = AutoNCP(ncp_units, latent_dim) 
        
        # LTC Layer
        # Input size is latent_dim (features from encoder)
        self.ltc = LTC(latent_dim, wiring, batch_first=True)
        
        # LTC with AutoNCP writes to output neurons which match latent_dim size.
        # So output of LTC is (B, T, latent_dim).
        
        # --- 3. Decoder ---
        # We need to upsample from (B, latent_dim) back to (B, 1, 64, 64)
        
        self.decoder_input_proj = nn.Linear(latent_dim, 512 * 4 * 4) # Start at 4x4 spatial
        
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Final projection
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
        
    def forward(self, x, hx=None):
        """
        x: (Batch, Seq_Len, Channels, H, W)
        hx: Initial state for LTC (optional)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Flatten time and batch for CNN encoder
        x_flat = x.view(batch_size * seq_len, C, H, W)
        
        # Encode
        features = self.encoder_backbone(x_flat) # (B*T, 512, 1, 1)
        features = features.view(batch_size * seq_len, -1) # (B*T, 512)
        
        kl_loss = 0.0
        
        if self.variational:
            latent, kl_loss = self.vae_encoder(features)
        else:
            latent = self.fc_encode(features) # (B*T, latent)
        
        # Unflatten for RNN
        latent_seq = latent.view(batch_size, seq_len, -1)
        
        # Run LTC
        # out: (Batch, Seq_Len, latent_dim)
        ltc_out, hx_new = self.ltc(latent_seq, hx)
        
        # Decode
        # We need to decode every time step
        rnn_flat = ltc_out.contiguous().view(batch_size * seq_len, -1)
        
        # Spatially decode
        decoder_in = self.decoder_input_proj(rnn_flat) # (B*T, 512*16)
        decoder_in = decoder_in.view(batch_size * seq_len, 512, 4, 4)
        
        pred_map = self.decoder(decoder_in) # (B*T, 1, 64, 64)
        
        # Reshape to (Batch, Seq_Len, 1, H, W)
        pred_map = pred_map.view(batch_size, seq_len, 1, 64, 64)
        
        if self.variational:
            return pred_map, kl_loss
            
        return pred_map

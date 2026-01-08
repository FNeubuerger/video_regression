import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()
        
        # We use bilinear upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Input channels to conv will be in_channels (after up) + skip_channels
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        """
        x1: Input from previous decoder layer
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # Handle padding issues if dimensions don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
             x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class VariationalBottleneck(nn.Module):
    """
    Probabilistic Bottleneck using Re-parameterization Trick.
    Encodes feature map into Normal Distribution N(mu, sigma).
    Sample z ~ N(mu, sigma) = mu + sigma * epsilon.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv_mu = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=10)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        if self.training:
            z = mu + std * eps
        else:
            # During inference, we can sample OR return mean
            # Ideally for Bayesian Eval we sample.
            # But let's support both via mode switch?
            # For now, let's always sample to enable uncertainty estimation.
            z = mu + std * eps
            
        # KL Divergence term (sum over spatial dims)
        # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        kl_loss = torch.mean(kl_loss) # Batch mean
        
        return z, kl_loss

class ResNetUNet(nn.Module):
    """
    U-Net with ResNet18 Encoder.
    """
    def __init__(self, n_channels=5, n_classes=1, variational=False):
        super(ResNetUNet, self).__init__()
        self.variational = variational
        
        # Load ResNet18
        self.base_model = resnet18(weights='IMAGENET1K_V1')

        
        # Adapt first layer
        if n_channels != 3:
            original_conv1 = self.base_model.conv1
            self.base_model.conv1 = nn.Conv2d(
                n_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            # Initialize new weights
            with torch.no_grad():
                self.base_model.conv1.weight[:, :3, :, :] = original_conv1.weight
                if n_channels > 3:
                     nn.init.kaiming_normal_(self.base_model.conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
        
        # Encoder Layers
        self.enc_conv1 = self.base_model.conv1 # -> H/2
        self.enc_bn1 = self.base_model.bn1
        self.enc_relu = self.base_model.relu
        self.enc_maxpool = self.base_model.maxpool # -> H/4
        
        self.enc_layer1 = self.base_model.layer1 # -> H/4 (64 ch)
        self.enc_layer2 = self.base_model.layer2 # -> H/8 (128 ch)
        self.enc_layer3 = self.base_model.layer3 # -> H/16 (256 ch)
        self.enc_layer4 = self.base_model.layer4 # -> H/32 (512 ch)
        
        # Variational Bottleneck
        if self.variational:
            self.bottleneck = VariationalBottleneck(512)
        
        # Decoder Layers
        # layer4 (512) -> layer3 (256)
        self.up1 = UpBlock(512, 256, 256) 
        # up1_out (256) -> layer2 (128)
        self.up2 = UpBlock(256, 128, 128)
        # up2_out (128) -> layer1 (64)
        self.up3 = UpBlock(128, 64, 64)
        # up3_out (64) -> conv1_relu (64) (H/2)
        self.up4 = UpBlock(64, 64, 64)
        
        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        # Support 5D input (batch, time, ch, h, w) -> take last frame
        if x.dim() == 5:
            x = x[:, -1, :, :, :]
            
        # Encoder
        x1 = self.enc_conv1(x)
        x1 = self.enc_bn1(x1)
        x1 = self.enc_relu(x1) # Skip 1: H/2, 64
        
        x2 = self.enc_maxpool(x1) 
        x2 = self.enc_layer1(x2) # Skip 2: H/4, 64
        
        x3 = self.enc_layer2(x2) # Skip 3: H/8, 128
        x4 = self.enc_layer3(x3) # Skip 4: H/16, 256
        x5 = self.enc_layer4(x4) # Bottleneck: H/32, 512
        
        # Variational
        kl_loss = 0.0
        if self.variational:
            x5, kl_loss = self.bottleneck(x5)
            
        # Decoder
        d5 = self.up1(x5, x4)
        d4 = self.up2(d5, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up4(d3, x1)
        
        out = self.final_up(d2)
        
        if self.variational:
             return out, kl_loss
        
        return out

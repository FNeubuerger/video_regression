import torch
import torch.nn as nn
import torchbnn as bnn
from torchvision.models import resnet18

class BayesianResNet(nn.Module):
    """
    Bayesian ResNet for temperature regression.
    Uses a standard ResNet18 backbone for feature extraction,
    but the final regression layers are Bayesian (Variational Inference).
    """

    def __init__(self, frame_shape, prior_mu=0.0, prior_sigma=0.1, out_features=4):
        """
        Initializes the Bayesian ResNet model.

        Parameters:
        - frame_shape: Tuple representing the shape of a single frame (channels, height, width) 
                       or (time, channels, height, width).
        - out_features: Number of output regression targets (default: 4 for sensors).
        """
        super(BayesianResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = resnet18(weights='IMAGENET1K_V1')
        
        # Determine channel dimension
        # The project convention for frame_shape is (Height, Width, Channels)
        # However, we also want to be robust if (Channels, Height, Width) is passed
        if len(frame_shape) == 3:
            if frame_shape[0] <= 4 and frame_shape[2] > 4: # Likely (C, H, W)
                in_channels = frame_shape[0]
            else: # Likely (H, W, C)
                in_channels = frame_shape[2]
        elif len(frame_shape) == 4: # (T, C, H, W) or (T, H, W, C)
             # Assume (T, C, H, W) if index 1 is small
             if frame_shape[1] <= 4:
                 in_channels = frame_shape[1]
             else:
                 in_channels = frame_shape[3]
        else:
             in_channels = 3 # Default fallback
        
        # Handle input channels != 3
        if in_channels != 3:
            original_conv1 = self.backbone.conv1
            new_conv1 = nn.Conv2d(
                in_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                if in_channels > 3:
                    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            self.backbone.conv1 = new_conv1
        
        # Get the number of features from the original fc layer
        num_features = self.backbone.fc.in_features
        
        # Remove the original fc layer
        self.backbone.fc = nn.Identity()
        
        # Define Bayesian Regression Head
        # We replace the standard Linear layers with BayesianLinear layers
        self.bayesian_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=num_features, out_features=512),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=512, out_features=out_features) # Output determined by arg
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor of shape (batch_size, channels, height, width) or
             (batch_size, time_steps, channels, height, width).

        Returns:
        - Output tensor of shape (batch_size, 1).
        """
        # If input has time dimension, take the last frame
        if x.dim() == 5:
            x = x[:, -1, :, :, :]  # Take last frame
        
        # Forward through ResNet backbone
        features = self.backbone(x)
        
        # Forward through Bayesian Head
        predictions = self.bayesian_head(features)
        
        # Calculate KL Divergence
        # Use torchbnn.BKLLoss to calculate KL divergence of the whole model or head
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)(self)
        
        return predictions, kl_loss

class BayesianCNNLSTM(nn.Module):
    """
    Bayesian CNN-LSTM for temporal temperature regression.
    Combines a ResNet backbone, an LSTM for temporal dynamics, 
    and a Bayesian regression head.
    """
    def __init__(self, frame_shape, hidden_size=128, prior_mu=0.0, prior_sigma=0.1):
        super(BayesianCNNLSTM, self).__init__()
        
        # 1. CNN Backbone (ResNet18)
        self.backbone = resnet18(weights='IMAGENET1K_V1')
        
        # Determine channel dimension
        # The project convention for frame_shape is (Height, Width, Channels)
        # However, we also want to be robust if (Channels, Height, Width) is passed
        if len(frame_shape) == 3:
            if frame_shape[0] <= 4 and frame_shape[2] > 4: # Likely (C, H, W)
                in_channels = frame_shape[0]
            else: # Likely (H, W, C)
                in_channels = frame_shape[2]
        elif len(frame_shape) == 4: # (T, C, H, W) or (T, H, W, C)
             if frame_shape[1] <= 4:
                 in_channels = frame_shape[1]
             else:
                 in_channels = frame_shape[3]
        else:
             in_channels = 3 # Default

        # Handle input channels != 3
        if in_channels != 3:
            original_conv1 = self.backbone.conv1
            new_conv1 = nn.Conv2d(
                in_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                if in_channels > 3:
                    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            self.backbone.conv1 = new_conv1
            
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # 2. LSTM
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, batch_first=True)
        
        # 3. Bayesian Head
        # Takes LSTM output and predicts temperature
        self.bayesian_head = nn.Sequential(
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=hidden_size, out_features=64),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=64, out_features=4) # Output 4
        )
        
    def forward(self, x):
        # x: (batch, time, channels, H, W)
        batch_size, time_steps, C, H, W = x.size()
        
        # CNN
        c_in = x.view(batch_size * time_steps, C, H, W)
        features = self.backbone(c_in)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features) # (batch, time, hidden)
        
        # Bayesian Head (applied to each time step)
        # Flatten time for efficiency
        lstm_out_flat = lstm_out.contiguous().view(batch_size * time_steps, -1)
        predictions = self.bayesian_head(lstm_out_flat)
        
        # Calculate KL Divergence
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)(self)
        
        # Reshape to (Batch, Time, 4)
        return predictions.view(batch_size, time_steps, 4), kl_loss

def convert_layer_to_bayesian(module, prior_mu, prior_sigma):
    """
    Recursively convert standard layers to Bayesian layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Replace Conv2d with BayesConv2d
            # Note: torchbnn.BayesConv2d signature: 
            # (prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
            new_layer = bnn.BayesConv2d(
                prior_mu=prior_mu, 
                prior_sigma=prior_sigma,
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=(child.bias is not None)
            )
            setattr(module, name, new_layer)
        elif isinstance(child, nn.Linear):
            # Replace Linear with BayesLinear
            new_layer = bnn.BayesLinear(
                prior_mu=prior_mu, 
                prior_sigma=prior_sigma,
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None)
            )
            setattr(module, name, new_layer)
        else:
            # Recursively convert children
            convert_layer_to_bayesian(child, prior_mu, prior_sigma)

class FullBayesianResNet(nn.Module):
    """
    Full Bayesian ResNet where ALL learnable layers (Conv2d, Linear) are Bayesian.
    """
    def __init__(self, frame_shape, prior_mu=0.0, prior_sigma=0.1):
        super(FullBayesianResNet, self).__init__()
        
        # Load standard ResNet18
        self.backbone = resnet18(weights=None) # No weights, we will initialize Bayesian layers
        
        # Determine channel dimension
        if len(frame_shape) == 3:
            if frame_shape[0] <= 4 and frame_shape[2] > 4: # Likely (C, H, W)
                in_channels = frame_shape[0]
            else: # Likely (H, W, C)
                in_channels = frame_shape[2]
        elif len(frame_shape) == 4: 
             if frame_shape[1] <= 4:
                 in_channels = frame_shape[1]
             else:
                 in_channels = frame_shape[3]
        else:
             in_channels = 3 
        
        # Handle input channels != 3
        if in_channels != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
        
        # Modify the final layer for regression (4 output)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 4)
        
        # Convert all layers to Bayesian
        convert_layer_to_bayesian(self.backbone, prior_mu, prior_sigma)

    def forward(self, x):
        # If input has time dimension, take the last frame
        if x.dim() == 5:
            x = x[:, -1, :, :, :]
            
        pred = self.backbone(x)
        
        # Calculate KL Divergence for all Bayesian layers
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)(self)
                
        return pred, kl_loss


    
class BayesianSpatialResNet(nn.Module):
    """
    Bayesian Spatial ResNet that outputs a temperature MAP (not a scalar).
    This enables spatial physics constraints like Convection (v . grad T).
    """
    def __init__(self, frame_shape, prior_mu=0.0, prior_sigma=0.1):
        super(BayesianSpatialResNet, self).__init__()
        
        # 1. Backbone (ResNet18)
        base_model = resnet18(weights='IMAGENET1K_V1')
        
        # Determine channel dimension
        if len(frame_shape) == 3:
            if frame_shape[0] <= 4 and frame_shape[2] > 4: # Likely (C, H, W)
                in_channels = frame_shape[0]
            else: # Likely (H, W, C)
                in_channels = frame_shape[2]
        elif len(frame_shape) == 4:
             if frame_shape[1] <= 4:
                 in_channels = frame_shape[1]
             else:
                 in_channels = frame_shape[3]
        else:
             in_channels = 3
        
        # Handle input channels != 3
        if in_channels != 3:
            original_conv1 = base_model.conv1
            new_conv1 = nn.Conv2d(
                in_channels, 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                if in_channels > 3:
                    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            base_model.conv1 = new_conv1
            
        # Remove FC and AvgPool to keep spatial dims
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        
        # 2. Bayesian Decoder
        # Maps features (512) to Temperature Map (1)
        # We use BayesConv2d to maintain uncertainty quantification in the spatial mapping
        self.decoder = nn.Sequential(
            bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            bnn.BayesConv2d(prior_mu=prior_mu, prior_sigma=prior_sigma, in_channels=128, out_channels=1, kernel_size=1)
        )
        
        # Upsampling to 4x4 or keeping at 2x2?
        # SpatialPhysicsCNNLSTM used a 4x4 map.
        # If input is 64x64, layer4 is 2x2.
        # Let's add an upsampling layer to get to 4x4 for better gradients.
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # x: (batch, channels, H, W) or (batch, time, channels, H, W)
        
        # If input has time dimension, we can process all frames or just the last one.
        # For "Spatial" models, we usually process single frames.
        # But if we want to do "Bayesian Convection PINN", we might need time for dT/dt?
        # Wait, "Spatial Convection" in BENCHMARK_PLAN uses "Steady-state Convection".
        # So single frame is fine.
        
        if x.dim() == 5:
            # If sequence, just take the last frame for now, 
            # OR process all frames if we want to support temporal sequences later.
            # Let's support both by merging batch and time.
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            has_time = True
        else:
            has_time = False
            
        features = self.cnn(x) # (N, 512, 2, 2)
        
        # Decode
        out = self.decoder(features) # (N, 1, 2, 2)
        
        # Upsample to 4x4
        out = self.upsample(out) # (N, 1, 4, 4)
        
        # Reshape back if time was merged
        if has_time:
            # Output map is (N, 1, 4, 4)
            # Reshape to (B, T, 1, 4, 4)
            out = out.view(B, T, 1, 4, 4)
        else:
            # Keep (B, 1, 4, 4)
            pass
            
        # Calculate KL Divergence for all Bayesian layers
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)(self)
        
        return out, kl_loss

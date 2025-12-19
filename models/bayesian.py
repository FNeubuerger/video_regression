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

    def __init__(self, frame_shape, prior_mu=0.0, prior_sigma=0.1):
        """
        Initializes the Bayesian ResNet model.

        Parameters:
        - frame_shape: Tuple representing the shape of a single frame (height, width, channels).
        - prior_mu: Mean of the prior distribution for weights.
        - prior_sigma: Standard deviation of the prior distribution for weights.
        """
        super(BayesianResNet, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = resnet18(weights='IMAGENET1K_V1')
        
        # Handle input channels != 3
        if frame_shape[2] != 3:
            original_conv1 = self.backbone.conv1
            new_conv1 = nn.Conv2d(
                frame_shape[2], 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                if frame_shape[2] > 3:
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
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=512, out_features=1)
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
        output = self.bayesian_head(features)
        return output.squeeze(-1)

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
        
        # Handle input channels != 3
        if frame_shape[2] != 3:
            original_conv1 = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                frame_shape[2], 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
        
        # Modify the final layer for regression (1 output)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)
        
        # Convert all layers to Bayesian
        convert_layer_to_bayesian(self.backbone, prior_mu, prior_sigma)

    def forward(self, x):
        # If input has time dimension, take the last frame
        if x.dim() == 5:
            x = x[:, -1, :, :, :]
            
        return self.backbone(x).squeeze(-1)

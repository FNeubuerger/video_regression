import torch
import torch.nn as nn
from torchvision.models import resnet18

class PhysicsCNNLSTM(nn.Module):
    """
    Physics-Informed CNN-LSTM that outputs a sequence of temperature predictions.
    This allows applying temporal consistency constraints (physics loss).
    """
    def __init__(self, frame_shape, time_steps, pretrained=True):
        super(PhysicsCNNLSTM, self).__init__()
        self.time_steps = time_steps
        
        # Backbone
        if pretrained:
            base_model = resnet18(weights='IMAGENET1K_V1')
            
            # Handle input channels != 3
            if frame_shape[2] != 3:
                original_conv1 = base_model.conv1
                new_conv1 = nn.Conv2d(
                    frame_shape[2], 
                    original_conv1.out_channels, 
                    kernel_size=original_conv1.kernel_size, 
                    stride=original_conv1.stride, 
                    padding=original_conv1.padding, 
                    bias=original_conv1.bias
                )
                with torch.no_grad():
                    if frame_shape[2] < 3:
                        new_conv1.weight[:, :, :, :] = original_conv1.weight[:, :frame_shape[2], :, :]
                    else:
                        new_conv1.weight[:, :3, :, :] = original_conv1.weight
                        if frame_shape[2] > 3:
                            nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                base_model.conv1 = new_conv1
            
            self.cnn = nn.Sequential(
                *list(base_model.children())[:-1], # Remove FC
                nn.Flatten()
            )
            cnn_out_size = 512
        else:
            # Simple custom CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(frame_shape[2], 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            cnn_out_size = 64

        # LSTM
        self.lstm = nn.LSTM(input_size=cnn_out_size, hidden_size=128, batch_first=True)
        
        # Regressor
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, time_steps, channels, H, W)
        batch_size, time_steps, C, H, W = x.size()
        
        # CNN Feature Extraction
        c_in = x.view(batch_size * time_steps, C, H, W)
        features = self.cnn(c_in)
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features) # (batch, time, hidden)
        
        # Regression on ALL time steps
        # We want to predict T for each frame to enforce smoothness
        predictions = self.fc(lstm_out) # (batch, time, 1)
        
        return predictions.squeeze(-1) # (batch, time)

class SpatialPhysicsCNNLSTM(nn.Module):
    """
    Advanced Physics-Informed Model that outputs a Spatio-Temporal Temperature Map.
    This allows applying spatial diffusion constraints (Laplacian).
    """
    def __init__(self, frame_shape, time_steps, pretrained=True):
        super(SpatialPhysicsCNNLSTM, self).__init__()
        self.time_steps = time_steps
        
        # Backbone (ResNet18)
        base_model = resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Handle input channels != 3
        if frame_shape[2] != 3:
            original_conv1 = base_model.conv1
            new_conv1 = nn.Conv2d(
                frame_shape[2], 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            with torch.no_grad():
                if frame_shape[2] < 3:
                    new_conv1.weight[:, :, :, :] = original_conv1.weight[:, :frame_shape[2], :, :]
                else:
                    new_conv1.weight[:, :3, :, :] = original_conv1.weight
                    if frame_shape[2] > 3:
                        nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            base_model.conv1 = new_conv1
        
        # Remove FC and AvgPool to keep spatial dims
        # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        # Output of layer4 is (512, H/32, W/32). For 64x64 input, this is (512, 2, 2).
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        
        # ConvLSTM or simple LSTM on flattened spatial?
        # For simplicity and speed, we'll use a 1x1 Conv to reduce channels, then treat as map
        # (Batch*Time, 512, 2, 2) -> (Batch*Time, 1, 2, 2)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1) # Predict temperature map directly from features
        )
        
        # Temporal smoothing via LSTM?
        # If we just use CNN, we lose temporal memory.
        # Let's insert a ConvLSTM or just a simple LSTM on the bottleneck.
        # Flatten (512, 2, 2) -> 2048 vector -> LSTM -> 2048 -> Reshape -> Decode
        self.lstm_input_size = 512 * 2 * 2
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=512, batch_first=True)
        
        # Decoder from LSTM hidden state to Spatial Map
        # Hidden (512) -> (1, 2, 2) map? No, let's upsample.
        # Let's aim for a 4x4 output map for better gradients
        self.spatial_decoder = nn.Sequential(
            nn.Linear(512, 16), # 4x4 map
            nn.ReLU()
        )

    def forward(self, x):
        # x: (batch, time_steps, channels, H, W)
        batch_size, time_steps, C, H, W = x.size()
        
        # CNN Feature Extraction
        c_in = x.view(batch_size * time_steps, C, H, W)
        features = self.cnn(c_in) # (B*T, 512, 2, 2)
        
        # Flatten for LSTM
        features_flat = features.view(batch_size, time_steps, -1)
        
        # LSTM
        lstm_out, _ = self.lstm(features_flat) # (batch, time, 512)
        
        # Decode to Spatial Map
        # We predict a 4x4 temperature map
        map_flat = self.spatial_decoder(lstm_out) # (batch, time, 16)
        temp_map = map_flat.view(batch_size, time_steps, 4, 4)
        
        return temp_map


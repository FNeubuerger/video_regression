import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    """
    CNNLSTM: A PyTorch model combining Convolutional Neural Networks (CNN) and 
    Long Short-Term Memory (LSTM) networks for video regression tasks.
    The model processes video sequences by first extracting spatial features 
    from individual frames using a CNN, and then capturing temporal dependencies 
    across frames using an LSTM. Finally, it performs regression using fully 
    connected layers.
    Attributes:
        time_steps (int): Number of frames in the video sequence.
        cnn (nn.Sequential): CNN module for spatial feature extraction.
        lstm (nn.LSTM): LSTM module for temporal feature processing.
        fc1 (nn.Linear): Fully connected layer for intermediate regression.
        fc2 (nn.Linear): Fully connected layer for final regression output.
    Methods:
        __init__(frame_shape, time_steps):
            Initializes the CNNLSTM model with the given frame shape and time steps.
        forward(x):
            Performs the forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of shape 
                    (batch_size, time_steps, channels, height, width).
                torch.Tensor: Output tensor of shape (batch_size, 1).
    """
    def __init__(self, frame_shape, time_steps):
        """
        Initializes the CNN-LSTM model for video regression.

        Parameters:
        - frame_shape: Tuple representing the shape of a single frame (height, width, channels).
        - time_steps: Number of frames in the video sequence.
        """
        super(CNNLSTM, self).__init__()
        self.time_steps = time_steps

        # Optimized CNN for feature extraction with batch normalization
        self.cnn = nn.Sequential(
            # First block - reduced channels for speed
            nn.Conv2d(frame_shape[2], 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Second block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Third block - smaller than original
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Global average pooling instead of regular flatten for efficiency
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # Calculate the flattened feature size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, frame_shape[2], frame_shape[0], frame_shape[1])
            cnn_output_size = self.cnn(dummy_input).shape[1]

        # Smaller LSTM for faster processing
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=64, batch_first=True)

        # Smaller fully connected layers for regression
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.fc2 = nn.Linear(32, 4) # Output 4 temperatures (one for each sensor)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
        - Output tensor of shape (batch_size, 1).
        """
        # Ensure the input tensor has 5 dimensions
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add a time_steps dimension if missing

        batch_size, time_steps, channels, height, width = x.size()

        # Reshape to process each frame through the CNN
        x = x.view(batch_size * time_steps, channels, height, width)
        cnn_features = self.cnn(x)

        # Reshape back to (batch_size, time_steps, cnn_output_size)
        cnn_features = cnn_features.view(batch_size, time_steps, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers with dropout
        x = torch.relu(self.fc1(lstm_out))
        x = self.dropout(x)
        output = self.fc2(x)

        return output.squeeze(-1)


class PretrainedCNNLSTM(nn.Module):
    """
    PretrainedCNNLSTM: A PyTorch module that combines a pretrained CNN for spatial feature extraction 
    and an LSTM for temporal processing, designed for video regression tasks.
    """

    def __init__(self, pretrained_cnn, frame_shape, time_steps):
        """
        Initializes the Pretrained CNN-LSTM model for video regression.

        Parameters:
        - pretrained_cnn: A pretrained CNN model (e.g., ResNet, EfficientNet).
        - frame_shape: Tuple representing the shape of a single frame (height, width, channels).
        - time_steps: Number of frames in the video sequence.
        """
        super(PretrainedCNNLSTM, self).__init__()
        self.time_steps = time_steps

        # Use the pretrained CNN for feature extraction
        # If input channels != 3, we need to modify the first layer
        if frame_shape[2] != 3:
            original_conv1 = list(pretrained_cnn.children())[0]
            new_conv1 = nn.Conv2d(
                frame_shape[2], 
                original_conv1.out_channels, 
                kernel_size=original_conv1.kernel_size, 
                stride=original_conv1.stride, 
                padding=original_conv1.padding, 
                bias=original_conv1.bias
            )
            # Initialize new weights (copy RGB weights to first 3 channels, random for others)
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                # Initialize remaining channels with small random values
                if frame_shape[2] > 3:
                    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            
            # Replace the first layer in the pretrained model
            # Note: We can't easily replace it in the sequential if we just take children
            # So we reconstruct the list of children
            layers = list(pretrained_cnn.children())
            layers[0] = new_conv1
            self.cnn = nn.Sequential(
                *layers[:-1],  # Remove the final classification layer
                nn.Flatten()
            )
        else:
            self.cnn = nn.Sequential(
                *list(pretrained_cnn.children())[:-1],  # Remove the final classification layer
                nn.Flatten()
            )

        # Calculate the flattened feature size after the pretrained CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, frame_shape[2], frame_shape[0], frame_shape[1])
            cnn_output_size = self.cnn(dummy_input).shape[1]

        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=128, batch_first=True)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 4) # Output 4 temperatures

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor of shape (batch_size, time_steps, channels, height, width).

        Returns:
        - Output tensor of shape (batch_size, 1).
        """
        # Ensure the input tensor has 5 dimensions
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add a time_steps dimension if missing

        batch_size, time_steps, channels, height, width = x.size()

        # Reshape to process each frame through the pretrained CNN
        x = x.view(batch_size * time_steps, channels, height, width)
        cnn_features = self.cnn(x)

        # Reshape back to (batch_size, time_steps, cnn_output_size)
        cnn_features = cnn_features.view(batch_size, time_steps, -1)

        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)

        # Take the output of the last time step
        lstm_out = lstm_out[:, -1, :]

        # Fully connected layers
        x = torch.relu(self.fc1(lstm_out))
        output = self.fc2(x)

        return output.squeeze(-1)

class SimpleResNet(nn.Module):
    """
    Simple ResNet for temperature regression from single images (no temporal component).
    """

    def __init__(self, frame_shape):
        """
        Initializes the Simple ResNet model for temperature regression.

        Parameters:
        - frame_shape: Tuple representing the shape of a single frame (height, width, channels).
        """
        super(SimpleResNet, self).__init__()
        
        # Load pretrained ResNet18
        from torchvision.models import resnet18
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
            # Initialize new weights
            with torch.no_grad():
                if frame_shape[2] < 3:
                    new_conv1.weight[:, :, :, :] = original_conv1.weight[:, :frame_shape[2], :, :]
                else:
                    new_conv1.weight[:, :3, :, :] = original_conv1.weight
                    if frame_shape[2] > 3:
                        nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
            self.backbone.conv1 = new_conv1
        
        # Get the number of features from the original fc layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final layer for regression
        self.backbone.fc = nn.Linear(num_features, 512)
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(512, 4) # Output 4 temperatures

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor of shape (batch_size, channels, height, width) or
             (batch_size, time_steps, channels, height, width) - will use last frame.

        Returns:
        - Output tensor of shape (batch_size, 1).
        """
        # If input has time dimension, take the last frame
        if x.dim() == 5:
            batch_size, time_steps, channels, height, width = x.size()
            x = x[:, -1, :, :, :]  # Take last frame
        
        # Forward through ResNet backbone
        features = self.backbone(x)
        features = self.dropout(features)
        output = self.regressor(features)
        return output.squeeze(-1)


class PretrainedCNN(nn.Module):
    """
    PretrainedCNN: A PyTorch module that uses a pretrained CNN for spatial feature extraction.
    """

    def __init__(self, pretrained_cnn, frame_shape):
        """
        Initializes the Pretrained CNN model for feature extraction.

        Parameters:
        - pretrained_cnn: A pretrained CNN model (e.g., ResNet, EfficientNet).
        - frame_shape: Tuple representing the shape of a single frame (height, width, channels).
        """
        super(PretrainedCNN, self).__init__()

        # Use the pretrained CNN for feature extraction
        self.cnn = nn.Sequential(
            *list(pretrained_cnn.children())[:-1],  # Remove the final classification layer
            nn.Flatten()
        )

        # Calculate the flattened feature size after the pretrained CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, frame_shape[2], frame_shape[0], frame_shape[1])
            # Ensure dummy input is on same device as model
            device = next(pretrained_cnn.parameters()).device
            dummy_input = dummy_input.to(device)
            self.feature_size = self.cnn(dummy_input).shape[1]

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - Output tensor of shape (batch_size, feature_size).
        """
        return self.cnn(x)
class SpatialResNet(nn.Module):
    """
    Spatial ResNet that outputs a temperature map instead of a scalar.
    Compatible with AdvancedBioHeatLoss if used in a sequence.
    """
    def __init__(self, frame_shape, output_map_size=(64, 64)):
        super(SpatialResNet, self).__init__()
        from torchvision.models import resnet18
        
        # Backbone
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
            
        # Remove FC and AvgPool to keep spatial dims
        # Output of layer4 is (512, H/32, W/32)
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        
        # Decoder to map
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Upsample(size=output_map_size, mode='bilinear', align_corners=False) # Upsample to target
        )

    def forward(self, x):
        # x: (batch, channels, H, W) or (batch, time, channels, H, W)
        if x.dim() == 5:
            # If sequence, process each frame
            batch, time, c, h, w = x.size()
            x = x.view(batch * time, c, h, w)
            features = self.cnn(x)
            out = self.decoder(features) # (B*T, 1, 4, 4)
            return out.view(batch, time, 4, 4)
        else:
            features = self.cnn(x)
            out = self.decoder(features)
            return out.squeeze(1) # (Batch, 4, 4)

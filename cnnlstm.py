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

        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(frame_shape[2], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Calculate the flattened feature size after CNN
        with torch.no_grad():
            dummy_input = torch.zeros(1, frame_shape[2], frame_shape[0], frame_shape[1])
            cnn_output_size = self.cnn(dummy_input).shape[1]

        # LSTM for temporal processing
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=128, batch_first=True)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

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

        # Fully connected layers
        x = torch.relu(self.fc1(lstm_out))
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
        self.fc2 = nn.Linear(64, 1)

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

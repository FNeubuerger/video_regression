import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM
from utils.dataset import TemperatureSequenceDataset
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
import argparse
from torch.cuda.amp import GradScaler, autocast


def train_model(model_instance, criterion_instance, optimizer_instance, data_dir="data", batch_size=32, num_epochs=1, learning_rate=0.001, model_save_path="models/cnn_lstm_model.pth", patience=5):
    # Set the learning rate for the optimizer
    for param_group in optimizer_instance.param_groups:
        param_group['lr'] = learning_rate
    
    # Define data transformations with optimization
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Reduced from 128x128 to 64x64 for faster processing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Normalize for better convergence
    ])

    # Create dataset with sequences for CNNLSTM
    dataset = TemperatureSequenceDataset(data_dir, sequence_length=3, transform=transform, image_size=(64, 64))  # Reduced sequence_length from 5 to 3
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loader with optimizations for large batches
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=8,  # More parallel data loading for large GPU
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4,  # Prefetch more batches
        drop_last=True  # Ensure consistent batch sizes for mixed precision
    )

    # Training loop with optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.to(device)
    
    # GPU memory optimization
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster computation
    
    # Mixed precision training
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Reduced gradient accumulation steps since we have larger batches
    accumulation_steps = 2
    
    best_loss = float('inf')  # Initialize best loss to infinity
    epochs_no_improve = 0  # Counter for epochs without improvement

    print(f"Training on device: {device}")
    print(f"Dataset size: {len(dataset)}, Training batches: {len(train_loader)}")

    for epoch in range(num_epochs):
        model_instance.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        
        optimizer_instance.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Mixed precision forward pass
            if scaler is not None:
                with autocast():
                    outputs = model_instance(images)
                    labels = labels.float()  # Ensure labels are float for regression
                    loss = criterion_instance(outputs, labels) / accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer_instance)
                    scaler.update()
                    optimizer_instance.zero_grad()
            else:
                # CPU training (no mixed precision)
                outputs = model_instance(images)
                labels = labels.float()
                loss = criterion_instance(outputs, labels) / accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer_instance.step()
                    optimizer_instance.zero_grad()

            running_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix(loss=(running_loss / (batch_idx + 1)))

        # Handle remaining gradients
        if (len(train_loader) % accumulation_steps) != 0:
            if scaler is not None:
                scaler.step(optimizer_instance)
                scaler.update()
            else:
                optimizer_instance.step()
            optimizer_instance.zero_grad()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Check for improvement
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            torch.save(model_instance.state_dict(), model_save_path)
            print(f"Model improved and saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        # Early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete. Model saved.")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a CNN-LSTM model.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for training.")
    args = parser.parse_args()

    num_epochs = args.epochs
    # Define model parameters - optimized for faster training with large GPU
    frame_shape = (64, 64, 3)  # Height, Width, Channels - reduced from 128x128
    time_steps = 3  # Number of frames in the video sequence - reduced from 5
    
    # Large batch size for GPU utilization
    batch_size = 128  # Increased from default 32

    # Initialize the model, loss function, and optimizer
    model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression
    # Use AdamW with higher learning rate for larger batches
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

    # Train the model with large batch size
    train_model(model, criterion, optimizer, num_epochs=num_epochs, batch_size=batch_size)
    # Download a pretrained ResNet model for medical imaging

    # Load a ResNet18 model pretrained on ImageNet
    pretrained_cnn = resnet18(weights='IMAGENET1K_V1')

    # Modify the final layer to fit the medical imaging task (e.g., regression)
    pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)  # Assuming single output for regression
    # Initialize the PretrainedCNNLSTM model using the pretrained ResNet
    pretrained_model = PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)

    # Define a new criterion and optimizer for the pretrained model
    pretrained_criterion = torch.nn.MSELoss()
    pretrained_optimizer = torch.optim.AdamW(pretrained_model.parameters(), lr=0.003, weight_decay=1e-4)

    # Train the pretrained model with large batch size
    train_model(pretrained_model, pretrained_criterion, pretrained_optimizer, num_epochs=num_epochs, batch_size=batch_size, model_save_path="models/pretrained_cnn_lstm_model.pth")
    
    pretrained_cnn2 = resnet18(weights='IMAGENET1K_V1')

    # Modify the final layer to fit the medical imaging task (e.g., regression)
    pretrained_cnn2.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)  # Assuming single output for regression
    # Initialize the PretrainedCNNLSTM model using the pretrained ResNet
    pretrained_model2 = PretrainedCNN(pretrained_cnn, frame_shape=frame_shape)

    # Define a new criterion and optimizer for the pretrained model
    pretrained_criterion2 = torch.nn.MSELoss()
    pretrained_optimizer2 = torch.optim.AdamW(pretrained_model2.parameters(), lr=0.003, weight_decay=1e-4)

    # Train the pretrained model with large batch size
    train_model(pretrained_model2, pretrained_criterion2, pretrained_optimizer2, num_epochs=num_epochs, batch_size=batch_size, model_save_path="models/pretrained_cnn_model.pth")

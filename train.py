from cnnlstm import CNNLSTM, PretrainedCNNLSTM, PretrainedCNN
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm
import argparse


def train_model(model_instance, criterion_instance, optimizer_instance, data_dir="data", batch_size=32, num_epochs=1, learning_rate=0.001, model_save_path="models/cnn_lstm_model.pth", patience=5):
    # Set the learning rate for the optimizer
    for param_group in optimizer_instance.param_groups:
        param_group['lr'] = learning_rate
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
    train_size = int(0.8 * len(dataset))  # Use 80% of the data for training
    val_size = len(dataset) - train_size
    train_dataset, _ = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.to(device)

    best_loss = float('inf')  # Initialize best loss to infinity
    epochs_no_improve = 0  # Counter for epochs without improvement

    for epoch in range(num_epochs):
        model_instance.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model_instance(images)
            labels = labels.float()  # Ensure labels are float for regression
            loss = criterion_instance(outputs, labels)

            # Backward pass and optimization
            optimizer_instance.zero_grad()
            loss.backward()
            optimizer_instance.step()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=(running_loss / (progress_bar.n + 1)))

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
    # Define model parameters
    frame_shape = (128, 128, 3)  # Height, Width, Channels
    time_steps = 5  # Number of frames in the video sequence

    # Initialize the model, loss function, and optimizer
    model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, criterion, optimizer, num_epochs=num_epochs)
    # Download a pretrained ResNet model for medical imaging

    # Load a ResNet18 model pretrained on ImageNet
    pretrained_cnn = resnet18(weights='IMAGENET1K_V1')

    # Modify the final layer to fit the medical imaging task (e.g., regression)
    pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)  # Assuming single output for regression
    # Initialize the PretrainedCNNLSTM model using the pretrained ResNet
    pretrained_model = PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)

    # Define a new criterion and optimizer for the pretrained model
    pretrained_criterion = torch.nn.MSELoss()
    pretrained_optimizer = torch.optim.Adam(pretrained_model.parameters(), lr=0.001)

    # Train the pretrained model
    train_model(pretrained_model, pretrained_criterion, pretrained_optimizer, num_epochs=num_epochs, model_save_path="models/pretrained_cnn_lstm_model.pth")
    
    pretrained_cnn2 = resnet18(weights='IMAGENET1K_V1')

    # Modify the final layer to fit the medical imaging task (e.g., regression)
    pretrained_cnn2.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)  # Assuming single output for regression
    # Initialize the PretrainedCNNLSTM model using the pretrained ResNet
    pretrained_model2 = PretrainedCNN(pretrained_cnn, frame_shape=frame_shape)

    # Define a new criterion and optimizer for the pretrained model
    pretrained_criterion2 = torch.nn.MSELoss()
    pretrained_optimizer2 = torch.optim.Adam(pretrained_model.parameters(), lr=0.001)

    # Train the pretrained model
    train_model(pretrained_model2, pretrained_criterion2, pretrained_optimizer2, num_epochs=num_epochs, model_save_path="models/pretrained_cnn_model.pth")

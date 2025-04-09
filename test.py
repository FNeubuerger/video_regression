import cv2
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from cnnlstm import CNNLSTM, PretrainedCNNLSTM
from tqdm import tqdm
from torchvision.models import resnet18

# Plot scatterplot and frame-by-frame predictions
def plot_results(true_temperatures, predicted_temperatures, frame_indices, save_path="results/plot.png"):
    """
    Plot scatterplot and frame-by-frame predictions of true vs predicted temperatures.

    Args:
        true_temperatures (list or np.ndarray): Ground truth temperature values.
        predicted_temperatures (list or np.ndarray): Predicted temperature values.
        frame_indices (list or np.ndarray): Indices of the frames corresponding to the temperatures.
        save_path (str): Path to save the plot.

    Functionality:
        - Plots a scatterplot comparing true and predicted temperatures.
        - Plots frame-by-frame temperature predictions for visualization.
        - Saves the plot to the specified path.
    """
    # Ensure true_temperatures and predicted_temperatures have the same length
    min_length = min(len(true_temperatures), len(predicted_temperatures))
    true_temperatures = true_temperatures[:min_length]
    predicted_temperatures = predicted_temperatures[:min_length]
    frame_indices = frame_indices[:min_length]

    # Scatterplot of true vs predicted temperatures
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(true_temperatures, predicted_temperatures, alpha=0.5)
    plt.xlabel("True Temperatures")
    plt.ylabel("Predicted Temperatures")
    plt.title("Scatterplot of True vs Predicted Temperatures")
    plt.grid()

    # Frame-by-frame temperature predictions
    plt.subplot(1, 2, 2)
    plt.plot(frame_indices, true_temperatures, label="True Temperatures", color="blue")
    plt.plot(frame_indices, predicted_temperatures, label="Predicted Temperatures", color="red")
    plt.xlabel("Frame Index")
    plt.ylabel("Temperature")
    plt.title("Frame-by-Frame Temperature Predictions")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def show_example_predictions(test_loader, model, device="cuda", num_examples=5, save_path="results/example_predictions.png"):
    """
    Display example test images with predicted and true temperatures.

    Args:
        test_loader (DataLoader): DataLoader for the test dataset.
        model: Trained model for prediction.
        device (str): Device to run the model on ("cuda" or "cpu").
        num_examples (int): Number of example images to display.
        save_path (str): Path to save the figure.

    Functionality:
        - Selects a few test images from the test_loader.
        - Predicts temperatures using the model.
        - Displays the images with predicted and true temperatures as annotations.
        - Saves the figure to the specified path.
    """
    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 5))

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).cpu().numpy()
            labels = labels.cpu().numpy()

            for i in range(min(num_examples - images_shown, len(images))):
                ax = axes[images_shown]
                img = images[i].cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
                img = (img - img.min()) / (img.max() - img.min())  # Normalize for display
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"True: {labels[i]:.2f}\nPred: {outputs[i]:.2f}")
                images_shown += 1

                if images_shown >= num_examples:
                    break
            if images_shown >= num_examples:
                break

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Main function
def evaluate(model, batch_size=32, data_dir="data", device="cuda", model_name="cnn_lstm_model"):
    """
    Evaluate a given model on a test dataset and calculate regression performance.
    Args:
        model: The model to be evaluated. It should have a `predict` method that accepts 
               a numpy array as input and returns predicted values.
    Functionality:
        - Applies transformations to the dataset, including resizing and converting to tensors.
        - Splits the dataset into training and testing subsets (80% training, 20% testing).
        - Loads the test dataset into a DataLoader for batch processing.
        - Predicts temperatures for each sample in the test dataset using the provided model.
        - Calculates the Root Mean Squared Error (RMSE) between true and predicted temperatures.
        - Plots the results and saves them to a results directory.
        - Writes evaluation metrics and predictions to a text file.
    Outputs:
        - Prints the RMSE to the console.
        - Saves a plot of the results.
        - Writes evaluation results, including RMSE and predictions, 
          to a text file in the `results` directory.
    Note:
        - Ensure the `data_dir` variable points to the correct dataset directory.
        - The `model_path` variable should be defined globally or passed to the function 
          to correctly name the results file.
    """
    
    # Define transformations for the test dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.ImageFolder(os.path.join(data_dir), transform=transform)
    train_size = int(0.8 * len(dataset))  # Use 80% of the data for training
    val_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Evaluation loop
    model.eval()
    true_temperatures = []
    predicted_temperatures = []
    frame_indices = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            true_temperatures.extend(labels.cpu().numpy())
            predicted_temperatures.extend(outputs.cpu().numpy())
            frame_indices.extend(range(idx * batch_size, (idx + 1) * batch_size))

    # Calculate regression performance (e.g., RMSE)
    rmse = np.sqrt(mean_squared_error(true_temperatures, predicted_temperatures))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot results
    plot_results(true_temperatures, predicted_temperatures, frame_indices, save_path=f"results/{model_name}_plot.png")
    # Show example predictions
    show_example_predictions(test_loader, model, device=device, num_examples=5, save_path=f"results/{model_name}_example_predictions.png")

if __name__ == "__main__":
    frame_shape = (128, 128, 3)  # Height, Width, Channels
    time_steps = 5  # Number of frames in the video sequence

    # Initialize the model, loss function, and optimizer
    model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
    model.load_state_dict(torch.load("models/cnn_lstm_model.pth"))  # Load the state dictionary
    model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    evaluate(model, device=device, model_name="cnn_lstm_model")

    pretrained_cnn = resnet18(weights='IMAGENET1K_V1')

    # Modify the final layer to fit the medical imaging task (e.g., regression)
    pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)  # Assuming single output for regression
    # Initialize the PretrainedCNNLSTM model using the pretrained ResNet
    pretrained_model = PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)

    pretrained_model.load_state_dict(torch.load("models/pretrained_cnn_lstm_model.pth"))  # Load the state dictionary
    pretrained_model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model.to(device)
    evaluate(pretrained_model, device=device, model_name="resnet_cnn_lstm_model")
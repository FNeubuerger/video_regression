"""
Quick Model Evaluation Script

A simplified version for quick performance checks and comparisons.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

from cnnlstm import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from dataset import TemperatureSequenceDataset


def quick_evaluate():
    """Quick evaluation of all available models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    dataset = TemperatureSequenceDataset("data", sequence_length=3, transform=transform, image_size=(64, 64))
    
    # Small test split for quick evaluation
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Use smaller subset for quick testing
    quick_test_size = min(1000, len(test_dataset))
    quick_test_dataset, _ = torch.utils.data.random_split(test_dataset, [quick_test_size, len(test_dataset) - quick_test_size])
    
    test_loader = DataLoader(quick_test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Quick evaluation on {len(quick_test_dataset)} samples")
    
    # Model configurations
    models_to_test = []
    frame_shape = (64, 64, 3)
    time_steps = 3
    
    # CNNLSTM
    if os.path.exists("models/cnn_lstm_model.pth"):
        try:
            model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
            model.load_state_dict(torch.load("models/cnn_lstm_model.pth", map_location=device))
            model.to(device)
            model.eval()
            models_to_test.append(("CNNLSTM", model))
            print("✓ Loaded CNNLSTM model")
        except Exception as e:
            print(f"✗ Failed to load CNNLSTM: {e}")
    
    # PretrainedCNNLSTM
    if os.path.exists("models/pretrained_cnn_lstm_model.pth"):
        try:
            pretrained_cnn = resnet18(weights='IMAGENET1K_V1')
            pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)
            model = PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
            model.load_state_dict(torch.load("models/pretrained_cnn_lstm_model.pth", map_location=device))
            model.to(device)
            model.eval()
            models_to_test.append(("PretrainedCNNLSTM", model))
            print("✓ Loaded PretrainedCNNLSTM model")
        except Exception as e:
            print(f"✗ Failed to load PretrainedCNNLSTM: {e}")
    
    # SimpleResNet
    if os.path.exists("models/simple_resnet_model.pth"):
        try:
            model = SimpleResNet(frame_shape=frame_shape)
            model.load_state_dict(torch.load("models/simple_resnet_model.pth", map_location=device))
            model.to(device)
            model.eval()
            models_to_test.append(("SimpleResNet", model))
            print("✓ Loaded SimpleResNet model")
        except Exception as e:
            print(f"✗ Failed to load SimpleResNet: {e}")
    
    # Evaluate each model
    results = []
    
    for model_name, model in models_to_test:
        print(f"\nEvaluating {model_name}...")
        
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Calculate metrics
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        results.append({
            'name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'true_values': true_values
        })
        
        print(f"Results for {model_name}:")
        print(f"  RMSE: {rmse:.3f}°C")
        print(f"  MAE:  {mae:.3f}°C")
        print(f"  R²:   {r2:.3f}")
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Quick comparison plot
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for i, result in enumerate(results):
            ax = axes[i]
            
            predictions = result['predictions']
            true_values = result['true_values']
            
            ax.scatter(true_values, predictions, alpha=0.6, s=10)
            
            # Perfect prediction line
            min_val = min(true_values.min(), predictions.min())
            max_val = max(true_values.max(), predictions.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            
            ax.set_xlabel('True Temperature (°C)')
            ax.set_ylabel('Predicted Temperature (°C)')
            ax.set_title(f'{result["name"]}\nRMSE: {result["rmse"]:.2f}°C, R²: {result["r2"]:.3f}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/quick_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nQuick comparison plot saved to results/quick_comparison.png")
        
        # Print summary
        print("\n" + "="*50)
        print("QUICK EVALUATION SUMMARY")
        print("="*50)
        
        best_model = min(results, key=lambda x: x['rmse'])
        print(f"Best model (lowest RMSE): {best_model['name']}")
        print(f"Best RMSE: {best_model['rmse']:.3f}°C")
    
    return results


if __name__ == "__main__":
    quick_evaluate()
"""
Model Evaluation and Comparison Script for Temperature Estimation

This script loads all trained models and evaluates their performance on the test dataset,
providing comprehensive metrics and visualizations for comparison.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
import pandas as pd
from tqdm import tqdm
import argparse
from scipy import stats
import sys

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.backbones import CNNLSTM, PretrainedCNNLSTM, SimpleResNet
from utils.dataset import TemperatureSequenceDataset
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and comparison class."""
    
    def __init__(self, data_dir="data", batch_size=64, device=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize dataset and data loader
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        self.dataset = TemperatureSequenceDataset(
            data_dir, 
            sequence_length=3, 
            transform=self.transform, 
            image_size=(64, 64)
        )
        
        # Split dataset (80% train, 20% test)
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        _, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size], 
            generator=torch.Generator().manual_seed(42)  # For reproducible splits
        )
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Evaluation setup complete:")
        print(f"- Device: {self.device}")
        print(f"- Test dataset size: {len(self.test_dataset)}")
        print(f"- Test batches: {len(self.test_loader)}")
    
    def load_model(self, model_name, model_path):
        """Load a trained model from checkpoint."""
        frame_shape = (64, 64, 3)
        time_steps = 3
        
        try:
            if model_name == "CNNLSTM":
                model = CNNLSTM(frame_shape=frame_shape, time_steps=time_steps)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                
            elif model_name == "PretrainedCNNLSTM":
                # Recreate the pretrained CNN
                pretrained_cnn = resnet18(weights='IMAGENET1K_V1')
                pretrained_cnn.fc = torch.nn.Linear(pretrained_cnn.fc.in_features, 1)
                model = PretrainedCNNLSTM(pretrained_cnn, frame_shape=frame_shape, time_steps=time_steps)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                
            elif model_name == "SimpleResNet":
                model = SimpleResNet(frame_shape=frame_shape)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                
            else:
                raise ValueError(f"Unknown model name: {model_name}")
            
            model.to(self.device)
            model.eval()
            print(f"Successfully loaded {model_name} from {model_path}")
            return model
            
        except Exception as e:
            print(f"Failed to load {model_name} from {model_path}: {e}")
            return None
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model and return predictions and metrics."""
        predictions = []
        true_values = []
        
        print(f"Evaluating {model_name}...")
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc=f"Testing {model_name}"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = model(images)
                
                # Store predictions and true values
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(labels.cpu().numpy())
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Calculate metrics
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(true_values, predictions)
        
        # Calculate percentage of predictions within certain thresholds
        abs_errors = np.abs(predictions - true_values)
        within_1c = np.mean(abs_errors <= 1.0) * 100
        within_2c = np.mean(abs_errors <= 2.0) * 100
        within_5c = np.mean(abs_errors <= 5.0) * 100
        
        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'correlation': correlation,
            'correlation_p_value': p_value,
            'within_1c': within_1c,
            'within_2c': within_2c,
            'within_5c': within_5c,
            'predictions': predictions,
            'true_values': true_values,
            'num_samples': len(predictions)
        }
        
        return metrics
    
    def plot_comparison(self, results, save_path="results/model_comparison.png"):
        """Create comprehensive comparison plots."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 12))
        
        # Number of models
        n_models = len(results)
        
        # 1. Scatter plots for each model
        for i, result in enumerate(results):
            plt.subplot(3, n_models, i + 1)
            
            predictions = result['predictions']
            true_values = result['true_values']
            
            plt.scatter(true_values, predictions, alpha=0.6, s=1)
            
            # Perfect prediction line
            min_val = min(true_values.min(), predictions.min())
            max_val = max(true_values.max(), predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            plt.xlabel('True Temperature (°C)')
            plt.ylabel('Predicted Temperature (°C)')
            plt.title(f'{result["model_name"]}\nR² = {result["r2_score"]:.3f}, RMSE = {result["rmse"]:.2f}°C')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Error distribution plots
        for i, result in enumerate(results):
            plt.subplot(3, n_models, n_models + i + 1)
            
            errors = result['predictions'] - result['true_values']
            
            plt.hist(errors, bins=50, alpha=0.7, density=True)
            plt.axvline(0, color='red', linestyle='--', linewidth=2)
            plt.xlabel('Prediction Error (°C)')
            plt.ylabel('Density')
            plt.title(f'{result["model_name"]}\nError Distribution\nMAE = {result["mae"]:.2f}°C')
            plt.grid(True, alpha=0.3)
        
        # 3. Metrics comparison bar plot
        plt.subplot(3, 1, 3)
        
        model_names = [r['model_name'] for r in results]
        metrics_to_plot = ['rmse', 'mae', 'r2_score']
        
        x = np.arange(len(model_names))
        width = 0.25
        
        for i, metric in enumerate(metrics_to_plot):
            values = [r[metric] for r in results]
            plt.bar(x + i * width, values, width, label=metric.upper(), alpha=0.8)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                plt.text(x[j] + i * width, v + max(values) * 0.01, f'{v:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Models')
        plt.ylabel('Metric Value')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, model_names)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Comparison plots saved to {save_path}")
    
    def create_metrics_table(self, results, save_path="results/metrics_comparison.csv"):
        """Create a detailed metrics comparison table."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Create DataFrame with metrics
        metrics_data = []
        for result in results:
            metrics_data.append({
                'Model': result['model_name'],
                'RMSE (°C)': f"{result['rmse']:.3f}",
                'MAE (°C)': f"{result['mae']:.3f}",
                'R² Score': f"{result['r2_score']:.3f}",
                'Correlation': f"{result['correlation']:.3f}",
                'Within 1°C (%)': f"{result['within_1c']:.1f}",
                'Within 2°C (%)': f"{result['within_2c']:.1f}",
                'Within 5°C (%)': f"{result['within_5c']:.1f}",
                'Test Samples': result['num_samples']
            })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(save_path, index=False)
        
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        print(f"Detailed metrics saved to {save_path}")
        
        return df
    
    def run_evaluation(self, model_configs):
        """Run complete evaluation for all models."""
        results = []
        
        for model_name, model_path in model_configs.items():
            if os.path.exists(model_path):
                model = self.load_model(model_name, model_path)
                if model is not None:
                    metrics = self.evaluate_model(model, model_name)
                    results.append(metrics)
                    
                    # Clean up GPU memory
                    del model
                    torch.cuda.empty_cache()
            else:
                print(f"Model file not found: {model_path}")
        
        if not results:
            print("No models could be evaluated!")
            return None
        
        # Create visualizations and reports
        self.plot_comparison(results)
        self.create_metrics_table(results)
        
        # Find best model by different criteria
        print("\n" + "="*50)
        print("BEST MODELS BY CRITERIA:")
        print("="*50)
        
        best_rmse = min(results, key=lambda x: x['rmse'])
        best_r2 = max(results, key=lambda x: x['r2_score'])
        best_mae = min(results, key=lambda x: x['mae'])
        
        print(f"Best RMSE: {best_rmse['model_name']} ({best_rmse['rmse']:.3f}°C)")
        print(f"Best R² Score: {best_r2['model_name']} ({best_r2['r2_score']:.3f})")
        print(f"Best MAE: {best_mae['model_name']} ({best_mae['mae']:.3f}°C)")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare trained models")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--models_dir", type=str, default="checkpoints", help="Directory containing model checkpoints")
    args = parser.parse_args()
    
    # Define model configurations
    model_configs = {
        "CNNLSTM": os.path.join(args.models_dir, "cnnlstm_model.pth"),
        "PretrainedCNNLSTM": os.path.join(args.models_dir, "pretrained_cnnlstm_model.pth"),
        "SimpleResNet": os.path.join(args.models_dir, "simple_resnet_model.pth")
    }
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(model_configs)
    
    if results:
        print(f"\nEvaluation complete! Results saved in 'results/' directory.")
        print(f"Evaluated {len(results)} models on {results[0]['num_samples']} test samples.")
    
    return results


if __name__ == "__main__":
    main()
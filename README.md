# Video Temperature Regression

This repository contains a comprehensive deep learning pipeline for temperature estimation from image sequences using multiple neural network architectures. The project implements three different models to compare temporal vs non-temporal approaches for temperature regression from thermal images.

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Training All Models](#training-all-models)
  - [Individual Model Training](#individual-model-training)
  - [Model Evaluation](#model-evaluation)
  - [Quick Evaluation](#quick-evaluation)
- [Project Structure](#project-structure)
- [Performance](#performance)
- [License](#license)

## Overview

This project focuses on temperature estimation from thermal image sequences using deep learning. It compares three different approaches:

1. **CNNLSTM** - Custom CNN-LSTM for temporal sequence modeling
2. **PretrainedCNNLSTM** - ResNet18 + LSTM leveraging pretrained features
3. **SimpleResNet** - Non-temporal ResNet18 baseline for single-frame prediction

The pipeline supports GPU acceleration with mixed precision training, early stopping, and comprehensive model comparison.

## Models

### CNNLSTM
- Custom CNN architecture with 3 convolutional layers (16→32→64 channels)
- LSTM with 64 hidden units for temporal modeling
- Processes sequences of 3 frames for temperature prediction

### PretrainedCNNLSTM  
- ResNet18 backbone pretrained on ImageNet
- LSTM temporal modeling on extracted features
- Combines transfer learning with sequence modeling

### SimpleResNet
- ResNet18 for single-frame temperature estimation
- Baseline model without temporal components
- Direct image-to-temperature regression

## Features

- **Multi-model training** with comprehensive comparison
- **GPU acceleration** with mixed precision training and optimized batch processing
- **Early stopping** to prevent overfitting with configurable patience
- **Comprehensive evaluation** with RMSE, MAE, and R² metrics
- **Data visualization** with training curves and prediction comparisons
- **Temperature sequence dataset** with automatic parsing from filename labels
- **Optimized data loading** with parallel workers and memory pinning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FNeubuerger/video_regression.git
   cd video_regression
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Dataset

The dataset consists of thermal image sequences organized in directories:
```
data/
├── sequence_1/
│   ├── frame_1_label_30.0.png
│   ├── frame_2_label_30.5.png
│   └── ...
├── sequence_2/
└── ...
```

Each image filename contains the frame number and temperature label, which is automatically parsed by the dataset loader.

## Usage

### Training All Models

Train all three models with early stopping and comprehensive comparison:

```bash
python train_all_models.py --epochs 50 --patience 10 --batch_size 128
```

Options:
- `--epochs`: Maximum training epochs (default: 50)
- `--patience`: Early stopping patience (default: 10)
- `--batch_size`: Training batch size (default: 128)
- `--models`: Specific models to train (`cnnlstm`, `pretrained_cnnlstm`, `simple_resnet`, or `all`)

### Individual Model Training

Train a single model with the basic training script:

```bash
python train.py --epochs 30 --batch_size 64 --model cnnlstm
```

### Model Evaluation

Comprehensive evaluation with metrics and visualizations:

```bash
python evaluate_models.py
```

This generates:
- Performance metrics (RMSE, MAE, R²)
- Training loss curves
- Prediction vs actual comparisons
- Model comparison plots

### Quick Evaluation

Fast evaluation of available trained models:

```bash
python quick_eval.py
```

### Generating Test Data

Generate synthetic thermal data for testing:

```bash
python generate_dummy_data.py
```

## Project Structure

```
video_regression/
├── cnnlstm.py              # Model architectures (CNNLSTM, PretrainedCNNLSTM, SimpleResNet)
├── dataset.py              # TemperatureSequenceDataset implementation
├── train.py                # Basic model training script
├── train_all_models.py     # Comprehensive training with early stopping
├── evaluate_models.py      # Model comparison and evaluation
├── quick_eval.py          # Quick model evaluation
├── generate_dummy_data.py  # Synthetic data generation
├── requirements.txt        # Python dependencies
├── data/                   # Thermal image sequences
│   ├── sequence_1/
│   ├── sequence_2/
│   └── ...
├── models/                 # Saved model checkpoints
│   ├── cnnlstm_model.pth
│   ├── pretrained_cnn_lstm_model.pth
│   └── simple_resnet_model.pth
└── results/               # Training plots and evaluation results
    ├── training_curves.png
    ├── model_comparison.png
    └── prediction_examples.png
```

## Performance

### Training Optimizations
- **Mixed precision training** for 2x speedup on modern GPUs
- **Large batch processing** (128 samples) for efficient GPU utilization
- **Optimized data loading** with 8 parallel workers and memory pinning
- **Early stopping** prevents overfitting and reduces training time

### Model Comparison
Based on validation performance:

| Model | Parameters | Training Time | RMSE | MAE | R² |
|-------|------------|---------------|------|-----|-----|
| CNNLSTM | ~50K | ~15 min | TBD | TBD | TBD |
| PretrainedCNNLSTM | ~11M | ~20 min | TBD | TBD | TBD |
| SimpleResNet | ~11M | ~10 min | TBD | TBD | TBD |

*Run training and evaluation to populate actual metrics*

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- **RAM**: 16GB+ recommended for large batch processing
- **Storage**: 2GB+ for dataset and models

## Code Style

This project follows [Black code formatting](https://black.readthedocs.io/en/stable/) with 88 character line limits. A pre-commit hook is recommended:

```bash
pip install pre-commit
pre-commit install
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the deep learning framework
- torchvision for pretrained models and transforms
- ResNet architecture from "Deep Residual Learning for Image Recognition"
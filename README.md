# video_regression

This repository contains a deep learning pipeline for training and evaluating CNN-LSTM models for video regression tasks. The project is implemented in Python using PyTorch and torchvision.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generating Dummy Data](#generating-dummy-data)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The project focuses on training CNN-LSTM models for video regression tasks. It includes support for training models from scratch or using pretrained CNN backbones (e.g., ResNet18). The pipeline supports early stopping and saving the best model during training.

## Features

- Train CNN-LSTM models for video regression.
- Use pretrained CNN backbones for feature extraction.
- Early stopping to prevent overfitting.
- Customizable training parameters (e.g., epochs, batch size, learning rate).
- Dummy data generation for testing purposes.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/video_regression.git
   cd video_regression
   ```
2. Create a virtual environment and install the required dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
## Usage

### Generating Dummy Data

    To generate dummy data for testing purposes, run:

    ```bash
    python generate_dummy_data.py
    ```

### Training
    To train the model, use the following command:

    ```bash
    python train.py --epochs <number_of_epochs>
    ```

    Replace `<number_of_epochs>` with the desired number of training epochs.



    This will create a dataset of synthetic video data for experimentation.

### Testing

    After training, you can test the model using:

    ```bash
    python test.py
    ```

This will evaluate the trained model on the test dataset.
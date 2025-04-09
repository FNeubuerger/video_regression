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
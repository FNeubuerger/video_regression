#!/bin/bash

# Configuration
# Benchmark Plan for Part 3: Bayesian / Uncertainty Quantification
# 1. Bayesian ResNet U-Net
# 2. Bayesian Latent LTC
# 3. Dense Metrics Evaluation

LOG_DIR="logs/part3"
mkdir -p $LOG_DIR

echo "Starting Part 3 Benchmarks: Bayesian Uncertainty"

# 1. Train Bayesian ResNet U-Net
# We need to create a training script for this or modify the existing one.
# Assuming we modify train_unet_sparse.py to accept --variational argument
# For now, let's create a placeholder script based on train_unet_sparse.py if it exists, or just plan it.
# We will use the existing train_unet_hybrid.py but add a variational flag if needed.
# Since we haven't updated the valid training scripts to handle the tuple return (pred, kl), 
# we need to do that first.

echo "Scripts need update for tuple return values. Please run 'python training/update_train_scripts.py' if it exists."

# 2. Benchmark Bayesian U-Net
echo "Running Bayesian U-Net Benchmark..."
.venv/bin/python training/train_unet_hybrid.py \
    --run_name unet_bayesian_v1 \
    --variational \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --beta_kl 0.01

# 3. Benchmark Bayesian LTC
echo "Running Bayesian LTC Benchmark..."
.venv/bin/python training/train_ltc.py \
    --run_name latent_ltc_bayesian_v1 \
    --model_type latent_ltc \
    --variational \
    --epochs 20 \
    --batch_size 4 \
    --lr 1e-3 \
    --beta_kl 0.01

echo "Benchmarks Launched."

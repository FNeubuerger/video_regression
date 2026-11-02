#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the Bayesian CNNLSTM training script
mkdir -p logs
python training/train_bayesian_cnnlstm.py --epochs 50 > logs/bayesian_cnnlstm.log 2>&1

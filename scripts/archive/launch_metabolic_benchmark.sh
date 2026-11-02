#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the metabolic bioheat training script
mkdir -p logs
python training/train_metabolic_bioheat.py --epochs 50 > logs/metabolic_bioheat.log 2>&1

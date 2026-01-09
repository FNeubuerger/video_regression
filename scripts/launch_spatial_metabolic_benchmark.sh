#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the spatial metabolic bioheat training script
mkdir -p logs
python training/train_spatial_metabolic_bioheat.py --epochs 50 > logs/spatial_metabolic_bioheat.log 2>&1

#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the spatial convection bioheat training script
mkdir -p logs
python training/train_spatial_convection_bioheat.py --epochs 50 > logs/spatial_convection_bioheat.log 2>&1

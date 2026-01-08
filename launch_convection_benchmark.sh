#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Run the convection bioheat training script
mkdir -p logs
python training/train_convection_bioheat.py --epochs 50 > logs/convection_bioheat.log 2>&1

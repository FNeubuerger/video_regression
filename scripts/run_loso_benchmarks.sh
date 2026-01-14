#!/bin/bash

# LOSO Cross-Validation Parallel Execution Script
# This script runs the Leave-One-Sequence-Out cross-validation for multiple models.

VENV_PATH="/mnt/data2/video_regression/.venv/bin/python"

# Standard set of models to benchmark
# Options: CNNLSTM, PretrainedCNNLSTM, SimpleResNet, PhysicsCNNLSTM, 
#          ConvectionBioheat, SpatialResNet, BayesianResNet, ConvLTC

MODELS=("CNNLSTM" "PhysicsCNNLSTM" "ConvectionBioheat" "SimpleResNet")
EPOCHS=10   # Reduced for full benchmark run
BATCH_SIZE=32

mkdir -p logs/loso

# To run a specific model, you can pass it as an argument: ./scripts/run_loso_benchmarks.sh BayesianResNet
if [ "$1" != "" ]; then
    MODELS=("$1")
fi

for MODEL in "${MODELS[@]}"; do
    echo "Starting LOSO for $MODEL..."
    
    # We run in background using tmux sessions for each model 
    # to avoid terminal blocking and allow for parallel execution
    SESSION_NAME="loso_${MODEL,,}"
    
    tmux new-session -d -s $SESSION_NAME \
        "$VENV_PATH evaluation/loso_cross_validation.py --model $MODEL --epochs $EPOCHS --batch_size $BATCH_SIZE --masked"
    
    echo "Launched $MODEL in tmux session: $SESSION_NAME"
done

echo "Use 'tmux ls' to monitor progress."
echo "Results will be saved in results/loso_*.csv"

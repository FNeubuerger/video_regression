#!/bin/bash

# LOSO Cross-Validation Parallel Execution Script
# This script runs the Leave-One-Sequence-Out cross-validation for multiple models.

VENV_PATH="/mnt/data2/video_regression/.venv/bin/python"

# Standard set of models to benchmark
# Options: CNNLSTM, PretrainedCNNLSTM, SimpleResNet, PhysicsCNNLSTM, 
#          ConvectionBioheat, SpatialResNet, BayesianResNet, 
#          FullBayesianResNet, BayesianCNNLSTM, ConvLTC

MODELS=("CNNLSTM" "PretrainedCNNLSTM" "PhysicsCNNLSTM" "ConvectionBioheat" "SimpleResNet" "SpatialResNet" "BayesianResNet" "FullBayesianResNet" "BayesianCNNLSTM" "ConvLTC")
EPOCHS=10   # Reduced for full benchmark run
BATCH_SIZE=32

mkdir -p logs/loso

# To run a specific model, you can pass it as an argument: ./scripts/run_loso_benchmarks.sh BayesianResNet
if [ "$1" != "" ]; then
    MODELS=("$1")
fi

i=0
for MODEL in "${MODELS[@]}"; do
    GPU=$((i % 2))
    SESSION_NAME="loso_${MODEL,,}"
    LOG_FILE="logs/loso/${SESSION_NAME}.log"
    echo "Starting LOSO for $MODEL on GPU $GPU (log: $LOG_FILE)..."

    # Kill session if it already exists to ensure a fresh run
    tmux kill-session -t $SESSION_NAME 2>/dev/null

    tmux new-session -d -s $SESSION_NAME \
        "CUDA_VISIBLE_DEVICES=$GPU $VENV_PATH evaluation/loso_cross_validation.py --model $MODEL --epochs $EPOCHS --batch_size $BATCH_SIZE --masked --no-wandb 2>&1 | tee $LOG_FILE"

    echo "Launched $MODEL in tmux session: $SESSION_NAME"
    i=$((i + 1))
done

echo "Use 'tmux ls' to monitor progress."
echo "Results will be saved in results/loso_*.csv"

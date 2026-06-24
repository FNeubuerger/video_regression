#!/bin/bash
# LOSO launcher with explicit GPU round-robin so we don't pile every
# training job onto GPU 0.

set -u
VENV_PATH="/mnt/data2/video_regression/.venv/bin/python"
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-32}
LOG_DIR=logs/loso

# Models that are confirmed end-to-end working (1-epoch smoke test passed).
# Add others here once their training-loop shape mismatches are fixed.
DEFAULT_MODELS=(
    "CNNLSTM"
    "PretrainedCNNLSTM"
    "SimpleResNet"
    "BayesianResNet"
    "FullBayesianResNet"
    "PhysicsCNNLSTM"
    "BayesianCNNLSTM"
)

if [ "$#" -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=("${DEFAULT_MODELS[@]}")
fi

mkdir -p "$LOG_DIR"

# Count available GPUs
N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$N_GPU" -lt 1 ]; then
    N_GPU=1
fi
echo "Distributing ${#MODELS[@]} models across ${N_GPU} GPU(s)."

i=0
for MODEL in "${MODELS[@]}"; do
    GPU=$((i % N_GPU))
    SESSION_NAME="loso_$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    LOG_FILE="$LOG_DIR/${SESSION_NAME}.log"
    CMD="CUDA_VISIBLE_DEVICES=${GPU} ${VENV_PATH} evaluation/loso_cross_validation.py \
        --model ${MODEL} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --masked --no-wandb \
        2>&1 | tee ${LOG_FILE}"
    tmux new-session -d -s "$SESSION_NAME" "$CMD"
    echo "  -> $SESSION_NAME on GPU $GPU (log: $LOG_FILE)"
    i=$((i + 1))
done

echo
echo "Monitor with: tmux ls; tmux attach -t loso_<model>"
echo "Results will land in results/loso_<MODEL>_masked.csv"

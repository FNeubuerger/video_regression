#!/bin/bash

# Regularized LOSO Cross-Validation Parallel Execution Script.
# Reruns LOSO with weight decay + train-time augmentation to close the
# mixed-split vs. LOSO generalization gap documented in the paper.

VENV_PATH="/mnt/data2/video_regression/.venv/bin/python"

MODELS=("SimpleResNet" "CNNLSTM")
EPOCHS=10
BATCH_SIZE=32
SEED=42
WEIGHT_DECAY=1e-4

mkdir -p logs/loso

if [ "$1" != "" ]; then
    MODELS=("$1")
fi

i=0
for MODEL in "${MODELS[@]}"; do
    GPU=$((i % 2))
    SESSION_NAME="loso_reg_${MODEL,,}"
    LOG_FILE="logs/loso/${SESSION_NAME}.log"
    echo "Starting regularized LOSO for $MODEL on GPU $GPU (log: $LOG_FILE)..."

    tmux kill-session -t $SESSION_NAME 2>/dev/null

    tmux new-session -d -s $SESSION_NAME \
        "CUDA_VISIBLE_DEVICES=$GPU $VENV_PATH evaluation/loso_cross_validation.py --model $MODEL --epochs $EPOCHS --batch_size $BATCH_SIZE --masked --augment --weight_decay $WEIGHT_DECAY --seed $SEED --no-wandb 2>&1 | tee $LOG_FILE"

    echo "Launched $MODEL in tmux session: $SESSION_NAME"
    i=$((i + 1))
done

echo "Use 'tmux ls' to monitor progress."
echo "Results will be saved in results/loso_*_seed${SEED}_aug_masked.csv"

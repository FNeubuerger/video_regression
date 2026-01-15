#!/bin/bash

# Physics Benchmarks Parallel Execution Script
# Launches each physics-informed model training in a separate tmux session on GPU 1.

VENV_PATH=".venv/bin/python"
LOG_DIR="logs/physics"
mkdir -p $LOG_DIR
EPOCHS=30

# List of jobs to run
# Format: "Script path" "Log filename" "Session name"
declare -a JOBS=(
    "training/train_bioheat.py bioheat.log phys_bioheat"
    "training/train_convection_bioheat.py convection.log phys_convec"
    "training/train_metabolic_bioheat.py metabolic.log phys_metabolic"
    "training/train_spatial_bioheat.py spatial_bioheat.log phys_sp_bio"
    "training/train_spatial_convection_bioheat.py spatial_convection.log phys_sp_conv"
    "training/train_spatial_metabolic_bioheat.py spatial_metabolic.log phys_sp_meta"
    "training/train_bayesian_cnnlstm.py bayesian_cnnlstm.log phys_bayes"
)

echo "Launching Physics Benchmarks on GPU 1 in parallel tmux sessions..."

for JOB in "${JOBS[@]}"; do
    # Split the string into variables
    read -r SCRIPT_PATH LOG_FILE SESSION_NAME <<< "$JOB"

    echo "Launching $SESSION_NAME ($SCRIPT_PATH)..."
    
    # Kill existing session if it exists to ensure fresh start
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    
    # Launch new session
    tmux new-session -d -s "$SESSION_NAME" \
        "CUDA_VISIBLE_DEVICES=1 $VENV_PATH $SCRIPT_PATH --epochs $EPOCHS > $LOG_DIR/$LOG_FILE 2>&1"
        
done

echo "All jobs launched."
echo "Use 'tmux ls' to monitor sessions."
echo "Logs are being written to $LOG_DIR/"

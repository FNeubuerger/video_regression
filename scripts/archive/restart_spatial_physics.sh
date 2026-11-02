#!/bin/bash

# Restart Spatial Physics Benchmarks (GPU 1)
# Launches correct spatial physics jobs in parallel tmux sessions.

VENV_PATH=".venv/bin/python"
LOG_DIR="logs/physics"
mkdir -p $LOG_DIR
EPOCHS=30

# List of jobs to run
# Format: "Script args" "Log filename" "Session name"
declare -a JOBS=(
    "training/train_spatial_bioheat.py --epochs $EPOCHS|spatial_bioheat.log|phys_sp_bio"
    "training/train_spatial_convection_bioheat.py --epochs $EPOCHS|spatial_convection.log|phys_sp_conv"
    "training/train_spatial_metabolic_bioheat.py --epochs $EPOCHS|spatial_metabolic.log|phys_sp_meta"
)

echo "Restarting Spatial Physics Benchmarks on GPU 1..."

for JOB in "${JOBS[@]}"; do
    IFS="|" read -r ARGS LOG_FILE SESSION_NAME <<< "$JOB"

    echo "Launching $SESSION_NAME..."
    
    # Kill existing session
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    
    # Launch new session
    # Note: Using GPU 1
    tmux new-session -d -s "$SESSION_NAME" \
        "CUDA_VISIBLE_DEVICES=1 $VENV_PATH $ARGS > $LOG_DIR/$LOG_FILE 2>&1"
        
done

echo "Spatial Physics jobs restarted."
echo "Use 'tmux ls' to monitor."

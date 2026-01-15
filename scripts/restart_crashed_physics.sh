#!/bin/bash

# Restart Crashed Physics Benchmarks
# Launches specific crashed physics models in parallel tmux sessions on GPU 1.

VENV_PATH=".venv/bin/python"
LOG_DIR="logs/physics"
mkdir -p $LOG_DIR
EPOCHS=30

# List of jobs to run (Crashed ones only)
# Format: "Script path" "Log filename" "Session name"
declare -a JOBS=(
    "training/train_bioheat.py bioheat_restart.log phys_bioheat_re"
    "training/train_convection_bioheat.py convection_restart.log phys_convec_re"
    "training/train_metabolic_bioheat.py metabolic_restart.log phys_meta_re"
    "training/train_spatial_bioheat.py spatial_bioheat_restart.log phys_sp_bio_re"
    "training/train_spatial_metabolic_bioheat.py spatial_metabolic_restart.log phys_sp_meta_re"
    "training/train_bayesian_cnnlstm.py bayesian_cnnlstm_restart.log phys_bayes_re"
)

echo "Restarting Crashed Physics Benchmarks on GPU 1..."

for JOB in "${JOBS[@]}"; do
    # Split the string into variables
    read -r SCRIPT_PATH LOG_FILE SESSION_NAME <<< "$JOB"

    echo "Launching $SESSION_NAME ($SCRIPT_PATH)..."
    
    # Kill existing session if it exists to ensure fresh start
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    
    # Launch new session
    # Using CUDA_VISIBLE_DEVICES=1 as in the original physics script
    tmux new-session -d -s "$SESSION_NAME" \
        "CUDA_VISIBLE_DEVICES=1 $VENV_PATH $SCRIPT_PATH --epochs $EPOCHS > $LOG_DIR/$LOG_FILE 2>&1"
        
done

echo "All crashed jobs restarted."
echo "Use 'tmux ls' to monitor sessions."
echo "Check new logs in $LOG_DIR/*_restart.log"

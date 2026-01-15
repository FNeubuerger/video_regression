#!/bin/bash

# Restart LTC Benchmarks (GPU 0)
# Launches LTC training jobs in parallel tmux sessions.

VENV_PATH=".venv/bin/python"
LOG_DIR="logs/ltc"
mkdir -p $LOG_DIR
EPOCHS=30

# List of jobs to run
# Format: "Script args" "Log filename" "Session name"
declare -a JOBS=(
    "training/train_ltc.py --run_name conv_ltc_benchmark --model_type conv_ltc --epochs $EPOCHS|conv_ltc_benchmark.log|ltc_conv"
    "training/train_ltc.py --run_name latent_ltc_benchmark --model_type latent_ltc --ncp_units 160 --epochs $EPOCHS|latent_ltc_benchmark.log|ltc_latent"
    "training/train_ltc.py --run_name latent_ltc_variational --model_type latent_ltc --variational --ncp_units 160 --epochs $EPOCHS|latent_ltc_variational.log|ltc_var"
)

echo "Restarting LTC Benchmarks on GPU 0..."

for JOB in "${JOBS[@]}"; do
    IFS="|" read -r ARGS LOG_FILE SESSION_NAME <<< "$JOB"

    echo "Launching $SESSION_NAME..."
    
    # Kill existing session
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null
    
    # Launch new session
    # Note: Using GPU 0
    tmux new-session -d -s "$SESSION_NAME" \
        "CUDA_VISIBLE_DEVICES=0 $VENV_PATH $ARGS > $LOG_DIR/$LOG_FILE 2>&1"
        
done

echo "LTC jobs restarted."
echo "Use 'tmux ls' to monitor."

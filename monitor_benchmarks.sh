#!/bin/bash

LOG_DIR="logs"
SESSION="video_regression_benchmarks"

# Function to handle Ctrl+C
trap "echo -e '\nExiting monitor.'; exit 0" SIGINT

while true; do
    clear
    echo "=== Video Regression Benchmark Monitor ==="
    echo "Time: $(date '+%H:%M:%S')"
    
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "Tmux Session: $SESSION [RUNNING]"
    else
        echo "Tmux Session: $SESSION [NOT FOUND]"
        echo "Benchmarks may have finished or crashed."
    fi
    
    echo "--------------------------------------------------------------------------------"
    printf "%-20s | %-20s | %s\n" "Job Name" "Log File" "Latest Status"
    echo "--------------------------------------------------------------------------------"
    
    # Function to get status
    get_status() {
        local logfile="$1"
        if [ -f "$logfile" ]; then
            # Get last non-empty line, truncate to 60 chars to show more info
            tail -n 10 "$logfile" | grep -v "^$" | tail -n 1 | cut -c 1-60
        else
            echo "Waiting for log..."
        fi
    }

    # Try to find WandB URL from logs
    WANDB_URL=$(grep -h "View project at" "$LOG_DIR"/*.log 2>/dev/null | head -n 1 | grep -o 'https://.*')
    if [ -z "$WANDB_URL" ]; then
        WANDB_URL="Waiting for sync..."
    fi
    
    printf "%-20s | %-20s | %s\n" "CNNLSTM" "cnnlstm.log" "$(get_status "$LOG_DIR/cnnlstm.log")"
    printf "%-20s | %-20s | %s\n" "Pretrained CNNLSTM" "pretrained_cnnlstm.log" "$(get_status "$LOG_DIR/pretrained_cnnlstm.log")"
    printf "%-20s | %-20s | %s\n" "Simple ResNet" "simple_resnet.log" "$(get_status "$LOG_DIR/simple_resnet.log")"
    printf "%-20s | %-20s | %s\n" "Physics CNNLSTM" "physics_cnnlstm.log" "$(get_status "$LOG_DIR/physics_cnnlstm.log")"
    printf "%-20s | %-20s | %s\n" "Ensemble" "ensemble.log" "$(get_status "$LOG_DIR/ensemble.log")"
    printf "%-20s | %-20s | %s\n" "Bayesian Head" "bayesian_head.log" "$(get_status "$LOG_DIR/bayesian_head.log")"
    printf "%-20s | %-20s | %s\n" "Full Bayesian" "full_bayesian.log" "$(get_status "$LOG_DIR/full_bayesian.log")"
    
    echo "--------------------------------------------------------------------------------"
    echo "WandB Dashboard: $WANDB_URL"
    echo "--------------------------------------------------------------------------------"
    echo "Press Ctrl+C to exit monitor"
    
    sleep 2
done

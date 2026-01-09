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
    
    # Table Header
    # Job Name (22) | Log File (28) | Start Time (19) | Status (Remaining)
    DIVIDER="------------------------------------------------------------------------------------------------------------------------------------"
    echo "$DIVIDER"
    printf "%-22s | %-28s | %-19s | %s\n" "Job Name" "Log File" "Start Time" "Latest Status"
    echo "$DIVIDER"
    
    # Function to get start time
    get_start_time() {
        local logfile="$1"
        if [ -f "$logfile" ]; then
            stat -c %y "$logfile" | cut -d '.' -f 1
        else
            echo "-"
        fi
    }

    # Function to get status line
    get_status_line() {
        local logfile="$1"
        if [ -f "$logfile" ]; then
            # Handle carriage returns from tqdm progress bars and strip ANSI colors
            # Cut to 80 chars to prevent wrapping
            tail -c 2000 "$logfile" | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | grep --color=never -v "^$" | tail -n 1 | cut -c 1-80
        else
            echo "Waiting for log..."
        fi
    }

    # Helper to print row
    print_row() {
        local name="$1"
        local log="$2"
        local path="$LOG_DIR/$log"
        printf "%-22s | %-28s | %-19s | %s\n" "$name" "$log" "$(get_start_time "$path")" "$(get_status_line "$path")"
    }

    # Try to find WandB URL from logs
    WANDB_URL=$(grep --color=never -h "View project at" "$LOG_DIR"/*.log 2>/dev/null | head -n 1 | grep -o 'https://.*')
    if [ -z "$WANDB_URL" ]; then
        WANDB_URL="Waiting for sync..."
    fi
    
    # Print Rows
    # Dynamically find all log files
    for logpath in "$LOG_DIR"/*.log; do
        if [ -f "$logpath" ]; then
            filename=$(basename "$logpath")
            # Generate a pretty name from filename: remove extension, replace _ with space, title case
            job_name=$(echo "$filename" | sed 's/\.log//' | sed 's/_/ /g' | awk '{for(i=1;i<=NF;i++)sub(/./,toupper(substr($i,1,1)),$i)}1')
            
            # Truncate job name if too long
            if [ ${#job_name} -gt 22 ]; then
                job_name="${job_name:0:19}..."
            fi
            
            print_row "$job_name" "$filename"
        fi
    done
    
    echo "$DIVIDER"
    echo "WandB Dashboard: $WANDB_URL"
    echo "$DIVIDER"
    echo "Press Ctrl+C to exit monitor"
    
    sleep 2
done

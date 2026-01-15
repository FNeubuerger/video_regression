#!/bin/bash

# Update Monitor Script to handle multiple Streams
# 1. Standard (Background PID)
# 2. Uncertainty (Background PID)
# 3. LTC (Tmux)
# 4. Physics (Tmux)
# 5. Bayesian Physics (Tmux)
# 6. Dense (Tmux)

LOG_DIR="logs"

# Function to handle Ctrl+C
trap "echo -e '\nExiting monitor.'; exit 0" SIGINT

while true; do
    clear
    echo "=== Video Regression Benchmark Monitor (6 Streams) ==="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "--------------------------------------------------------"

    # Function to print job status
    print_job() {
        local name="$1"
        local logfile="$2"
        local type="$3" # "BG" or "TMUX"
        
        printf "%-25s | " "$name"
        
        if [ ! -f "$logfile" ]; then
             printf "Waiting for log...\n"
             return
        fi
        
        # Get last meaningful line (stripping tqdm /r)
        # Use python buffering workaround if needed, but usually tail works
        # Grep -v empty lines, wandb lines
        local status=$(tail -c 2000 "$logfile" | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*m//g' | grep -v "^$" | grep -v "wandb" | tail -n 1 | cut -c 1-80)
        
        if [ -z "$status" ]; then
            printf "Log found, waiting for output...\n"
        else
            printf "%s\n" "$status"
        fi
    }

    echo "--- 1. Standard Stream ---"
    print_job "Standard (CNNLSTM)" "logs/retrain/restart_main.log" "BG"
    
    echo -e "\n--- 2. Uncertainty Stream ---"
    print_job "Ensemble (Uncertainty)" "logs/retrain/restart_uncertainty.log" "BG"

    echo -e "\n--- 3. Dynamics Stream (LTC) ---"
    print_job "Latent LTC" "logs/ltc/latent_ltc_benchmark.log" "TMUX"
    print_job "Conv LTC" "logs/ltc/conv_ltc_benchmark.log" "TMUX"

    echo -e "\n--- 4. Physics Stream (Non-Bayesian) ---"
    print_job "Scalar Bioheat" "logs/physics/bioheat.log" "TMUX"
    print_job "Scalar Convection" "logs/physics/convection.log" "TMUX"
    print_job "Scalar Metabolic" "logs/physics/metabolic.log" "TMUX"
    print_job "Spatial Bioheat" "logs/physics/spatial_bioheat.log" "TMUX"
    print_job "Spatial Convection" "logs/physics/spatial_convection.log" "TMUX"
    print_job "Spatial Metabolic" "logs/physics/spatial_metabolic.log" "TMUX"
    
    echo -e "\n--- 5. Bayesian Physics Stream ---"
    print_job "Bayesian PINN" "logs/physics/bayesian_pinn.log" "TMUX"
    print_job "Bayesian Convection" "logs/physics/bayesian_convection.log" "TMUX"
    print_job "Bayesian Metabolic" "logs/physics/bayesian_metabolic.log" "TMUX"
    print_job "Bayesian Spatial" "logs/physics/bayesian_spatial_convection.log" "TMUX"
    print_job "Bayesian CNNLSTM" "logs/retrain/restart_physics.log" "BG"

    echo -e "\n--- 6. Dense Prediction Stream (U-Net) ---"
    print_job "Standard U-Net" "logs/unet/unet_standard.log" "TMUX"
    print_job "Variational U-Net" "logs/unet/unet_variational.log" "TMUX"
    print_job "Hybrid U-Net" "logs/unet/unet_hybrid.log" "TMUX"

    echo -e "\n--------------------------------------------------------"
    echo "Active Tmux Sessions:"
    tmux ls 2>/dev/null | grep -E "phys|ltc|unet|bayes" | head -n 12

    sleep 5
done

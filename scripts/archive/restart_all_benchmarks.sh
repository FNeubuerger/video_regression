#!/bin/bash
# Master Benchmark Restart Script
# Date: 2026-02-02

WORKSPACE="/mnt/data2/video_regression"
VENV="$WORKSPACE/.venv"

# 1. Restart Bayesian/Physics models (dataset error fix)
echo "Restarting Bayesian/Physics models..."
bash "$WORKSPACE/scripts/restart_failed_due_to_dataset.sh"

# 2. Relaunch failed legacy/ensemble/physics models
echo "Relaunching failed legacy/ensemble/physics models..."
bash "$WORKSPACE/scripts/relaunch_failed_benchmarks.sh"

# 3. Restart ConvLTC and LTC UNet benchmarks
echo "Restarting ConvLTC and LTC UNet benchmarks..."
bash "$WORKSPACE/scripts/launch_conv_ltc_benchmark.sh"
bash "$WORKSPACE/scripts/launch_ltc_benchmark.sh"

# 4. Restart Spatial Convection Bioheat benchmark
echo "Restarting Spatial Convection Bioheat benchmark..."
bash "$WORKSPACE/scripts/launch_spatial_convection_benchmark.sh"

# 5. Restart U-Net stream (if script exists)
if [ -f "$WORKSPACE/scripts/launch_unet_benchmark.sh" ]; then
    echo "Restarting U-Net benchmark..."
    bash "$WORKSPACE/scripts/launch_unet_benchmark.sh"
else
    echo "U-Net launch script not found. Please add launch_unet_benchmark.sh if needed."
fi

echo "All restart commands submitted."
echo "Monitor with: tmux ls && tmux attach -t <session_name>"

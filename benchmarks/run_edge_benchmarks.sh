#!/bin/bash

# Exit on error
set -e

echo "========================================================"
echo "       Edge Deployment Benchmark Suite"
echo "========================================================"

PYTHON_EXEC="/mnt/data2/video_regression/.venv/bin/python"

# 1. Convert Models to ONNX (Force Regen)
echo "[1/4] Cleaning up old ONNX models..."
rm -rf models/onnx
mkdir -p models/onnx

echo "[1/4] Converting models to ONNX..."
$PYTHON_EXEC benchmarks/convert_to_onnx.py

# 2. Run PyTorch Benchmarks
echo "[2/4] Running PyTorch Benchmarks..."
$PYTHON_EXEC benchmarks/benchmark_deployment.py --frames 2 --mc-samples 10 --output results/benchmark_pytorch.csv

# 3. Run ONNX Benchmarks
echo "[3/4] Running ONNX Benchmarks..."
$PYTHON_EXEC benchmarks/benchmark_deployment.py --frames 2 --onnx --mc-samples 10 --output results/benchmark_onnx.csv

# 4. Merge and Analyze
echo "[4/4] Merging Results and Generating Plots..."
$PYTHON_EXEC benchmarks/merge_results.py
$PYTHON_EXEC benchmarks/analyze_results.py

echo "========================================================"
echo "       Benchmark Complete!"
echo "========================================================"
echo "Results: results/edge_benchmark_results.csv"
echo "Plots:   results/plots/"

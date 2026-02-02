"""
Unified Evaluation Script for Video Temperature Regression

This script runs evaluation for all models (Deterministic and Bayesian)
and generates consolidated results for the paper.
"""

import os
import subprocess
import json
import pandas as pd
import argparse
import sys
import concurrent.futures

def run_command(command, description):
    # Use sys.executable to ensure we use the same python interpreter
    python_exe = sys.executable
    if "python " in command:
        command = command.replace("python ", f"{python_exe} ")
    
    print(f"\n{'='*80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*80}\n")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--samples", type=int, default=20, help="Number of MC samples for Bayesian models")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    parser.add_argument("--force", action="store_true", help="Force re-evaluation of all models")
    parser.add_argument("--jobs", type=int, default=2, help="Number of parallel evaluation jobs")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("results/uncertainty_eval", exist_ok=True)

    # 1. Deterministic Evaluation
    # This generates results/metrics_comparison.csv and results/model_comparison.png
    det_command = f"python evaluation/evaluate_models.py --batch_size {args.batch_size}"
    run_command(
        det_command,
        "Deterministic Models Evaluation"
    )

    # 2. Bayesian/Uncertainty Evaluations
    bayesian_models = [
        ("BayesianResNet", "checkpoints/bayesian_resnet.pth"),
        ("BayesianCNNLSTM", "models/bayesian_cnnlstm.pth"),
        ("BayesianSpatialResNet", "models/bayesian_convection_pinn.pth"),
        ("LatentLTC_UNet", "checkpoints/ltc_unet/best_model.pth"),
        ("ResNetUNet", "checkpoints/unet_hybrid/best_model.pth"),
        ("BayesianPINN", "models/bayesian_pinn.pth"),
    ]

    # Add masked variants if they exist
    masked_bayesian_models = []
    for model_name, checkpoint in bayesian_models:
        # Check standard checkpoints
        masked_checkpoint = checkpoint.replace("models/", "models/masked/").replace("checkpoints/", "checkpoints/masked/")
        if os.path.exists(masked_checkpoint):
            masked_bayesian_models.append((f"{model_name}_masked", masked_checkpoint))
    
    available_tasks = []
    for model_name, checkpoint in bayesian_models + masked_bayesian_models:
        if os.path.exists(checkpoint):
            available_tasks.append((model_name, checkpoint))
        else:
            if "_masked" not in model_name: # Don't warn for missing masked variants
                print(f"Skipping {model_name}: Checkpoint {checkpoint} not found.")

    if available_tasks:
        print(f"\nRunning {len(available_tasks)} uncertainty evaluations with {args.jobs} parallel jobs...")
        
        # Determine number of GPUs
        try:
            gpu_info = subprocess.check_output("nvidia-smi -L", shell=True).decode()
            num_gpus = len(gpu_info.strip().split('\n'))
        except:
            num_gpus = 1
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = []
            for i, (model_name, checkpoint) in enumerate(available_tasks):
                gpu_id = i % num_gpus
                cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python evaluation/comprehensive_uncertainty_eval.py --model {model_name} --checkpoint {checkpoint} --samples {args.samples} --batch_size {args.batch_size}"
                if args.force:
                    cmd += " --force"
                
                futures.append(executor.submit(run_command, cmd, f"Uncertainty Evaluation for {model_name} (GPU {gpu_id})"))
            
            for future in concurrent.futures.as_completed(futures):
                future.result()

    # 3. Generate Consolidated Tables
    # This script reads results/uncertainty_eval/*.json and produces results/tables/*.tex
    if os.path.exists("evaluation/generate_tables.py"):
        run_command(
            "python evaluation/generate_tables.py",
            "Consolidating LaTeX Tables and LOSO Results"
        )

    # 4. (Optional) Run Visualization comparison
    if os.path.exists("paper/viz_performance.py"):
        run_command(
            "python paper/viz_performance.py",
            "Generating Final Performance Visualization"
        )
        
    # 5. Run LOSO Visualization
    if os.path.exists("evaluation/visualize_loso.py"):
        run_command(
            "python evaluation/visualize_loso.py",
            "Generating LOSO Visualization Plots"
        )

    print("\n" + "="*80)
    print("UNIFIED EVALUATION COMPLETED")
    print("Check 'results/' directory for metrics, plots, and tables.")
    print("="*80)

if __name__ == "__main__":
    main()

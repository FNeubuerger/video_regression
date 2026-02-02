# evaluation/evaluate_phase2_completed.py
import os
import argparse
import sys

# Ensure this script can find the models package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluation.evaluate_models import ModelEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate completed Phase 2 models")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Define paths to completed models
    # Using specific files identified in the audit
    model_configs = {
        "CNNLSTM": "models/cnnlstm_model.pth",
        "PretrainedCNNLSTM": "models/pretrained_cnnlstm_model.pth",
        "SimpleResNet": "models/simple_resnet_model.pth",
        "BioheatPINN": "models/bioheat_pinn_model.pth",
        "MetabolicBioheat": "models/metabolic_bioheat_model.pth",
        "SpatialBioheat": "models/spatial_bioheat_resnet.pth",
        "SpatialMetabolic": "models/spatial_metabolic_bioheat_resnet.pth",
        "SpatialConvection": "models/spatial_convection_bioheat_resnet.pth"
        # Skipping ConvectionBioheat (Standard) as it is restarting/running
    }

    # Verify paths exist
    valid_configs = {}
    print("Checking model checkpoints:")
    for name, path in model_configs.items():
        if os.path.exists(path):
            print(f"  [OK] {name}: {path}")
            valid_configs[name] = path
        else:
            print(f"  [MISSING] {name}: {path}")

    if not valid_configs:
        print("No valid models found to evaluate.")
        return

    print(f"\nEvaluating {len(valid_configs)} models...")
    
    evaluator = ModelEvaluator(
        data_dir="data/level1_cropped",
        batch_size=args.batch_size
    )
    
    # Run evaluation
    results = evaluator.run_evaluation(valid_configs)
    
    print("\nEvaluation Summary:")
    for res in results:
        print(f"{res['model_name']}: RMSE={res['rmse']:.4f}, MAE={res['mae']:.4f}, R2={res['r2_score']:.4f}")

if __name__ == "__main__":
    main()

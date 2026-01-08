import os
import json
import pandas as pd
import glob
import argparse

def generate_tables(results_dir="results/uncertainty_eval", output_dir="results/tables"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load all JSON results
    json_files = glob.glob(os.path.join(results_dir, "*_metrics.json"))
    data = []
    
    for f in json_files:
        with open(f, 'r') as file:
            metrics = json.load(file)
            # Extract model name from filename
            model_name = os.path.basename(f).replace("_metrics.json", "")
            metrics['Model'] = model_name
            data.append(metrics)
            
    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data)
    
    # 2. Define Categories
    # Map filenames to display names and categories
    # Based on BENCHMARK_PLAN.md
    
    model_map = {
        # Temporal
        "cnnlstm_model": ("CNNLSTM", "Temporal"),
        "pretrained_cnnlstm_model": ("Pretrained CNNLSTM", "Temporal"),
        "physics_cnnlstm_model": ("Physics CNNLSTM", "Temporal"),
        "advanced_bioheat_model": ("Bioheat PINN", "Temporal"),
        "convection_bioheat_model": ("Convection Bioheat", "Temporal"),
        "metabolic_bioheat_model": ("Metabolic Bioheat", "Temporal"),
        
        # Spatial
        "simple_resnet_model": ("Simple ResNet", "Spatial"),
        "spatial_bioheat_model": ("Spatial Bioheat", "Spatial"),
        "spatial_convection_model": ("Spatial Convection", "Spatial"),
        "spatial_metabolic_model": ("Spatial Metabolic", "Spatial"),
        
        # Uncertainty
        "Ensemble": ("Ensemble", "Uncertainty"),
        "bayesian_resnet": ("Bayesian Head", "Uncertainty"),
        "full_bayesian_resnet": ("Full Bayesian", "Uncertainty"),
        "bayesian_pinn": ("Bayesian PINN", "Uncertainty"),
        "bayesian_cnnlstm": ("Bayesian CNNLSTM", "Uncertainty"),
        "bayesian_metabolic_pinn": ("Bayesian Metabolic PINN", "Uncertainty"),
        "bayesian_spatial_convection": ("Bayesian Spatial Convection", "Uncertainty")
    }
    
    # Apply mapping
    def get_display_name(name):
        return model_map.get(name, (name, "Other"))[0]
        
    def get_category(name):
        return model_map.get(name, (name, "Other"))[1]
        
    df['Display Name'] = df['Model'].apply(get_display_name)
    df['Category'] = df['Model'].apply(get_category)
    
    # Reorder columns
    cols = ['Category', 'Display Name', 'MSE', 'MAE', 'RMSE', 'NLL', 'PICP_95', 'MPIW_95']
    df = df[cols]
    
    # Sort by Category and RMSE
    df = df.sort_values(by=['Category', 'RMSE'])
    
    # 3. Generate Tables
    
    # Main Summary Table
    print("\n=== Comprehensive Results ===")
    print(df.to_markdown(index=False, floatfmt=".4f"))
    df.to_csv(os.path.join(output_dir, "comprehensive_results.csv"), index=False)
    
    # Category Tables
    for category in df['Category'].unique():
        cat_df = df[df['Category'] == category].drop(columns=['Category'])
        print(f"\n=== {category} Models ===")
        print(cat_df.to_markdown(index=False, floatfmt=".4f"))
        cat_df.to_csv(os.path.join(output_dir, f"{category.lower()}_results.csv"), index=False)
        
        # LaTeX Export
        latex_path = os.path.join(output_dir, f"{category.lower()}_results.tex")
        cat_df.to_latex(latex_path, index=False, float_format="%.4f", caption=f"{category} Model Comparison")
        
    print(f"\nTables saved to {output_dir}")

if __name__ == "__main__":
    generate_tables()

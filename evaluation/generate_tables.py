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
            # Standardize keys to uppercase for the consolidated table
            standardized_metrics = {}
            for k, v in metrics.items():
                standardized_metrics[k.upper()] = v
            
            # Extract model name from filename
            model_name = os.path.basename(f).replace("_metrics.json", "")
            standardized_metrics['Model'] = model_name
            data.append(standardized_metrics)
            
    if not data:
        print("No results found.")
        return

    df = pd.DataFrame(data)
    
    # 2. Define Categories
    # Map filenames to display names and categories
    # Based on BENCHMARK_PLAN.md
    
    model_map = {
        # Temporal
        "cnnlstm": ("CNNLSTM", "Temporal"),
        "pretrained_cnnlstm": ("Pretrained CNNLSTM", "Temporal"),
        "physics_cnnlstm": ("Physics CNNLSTM", "Temporal"),
        "advanced_bioheat": ("Bioheat PINN", "Temporal"),
        "convection_bioheat": ("Convection Bioheat", "Temporal"),
        "metabolic_bioheat": ("Metabolic Bioheat", "Temporal"),
        
        # Spatial
        "simple_resnet": ("Simple ResNet", "Spatial"),
        "spatial_bioheat": ("Spatial Bioheat", "Spatial"),
        "spatial_convection": ("Spatial Convection", "Spatial"),
        "spatial_metabolic": ("Spatial Metabolic", "Spatial"),
        
        # Uncertainty
        "Ensemble": ("Ensemble", "Uncertainty"),
        "bayesian_resnet": ("Bayesian Head", "Uncertainty"),
        "full_bayesian_resnet": ("Full Bayesian", "Uncertainty"),
        "bayesian_pinn": ("Bayesian PINN", "Uncertainty"),
        "bayesian_pinn": ("Bayesian PINN", "Uncertainty"),
        "bayesian_cnnlstm": ("Bayesian CNNLSTM", "Uncertainty"),
        "bayesian_metabolic_pinn": ("Bayesian Metabolic PINN", "Uncertainty"),
        "bayesian_spatial_convection": ("Bayesian Spatial Convection", "Uncertainty")
    }
    
    # Apply mapping
    def get_display_name(name):
        is_masked = "_masked" in name or "masked_" in name or "model_masked" in name
        base_name = name.replace("_masked", "").replace("masked_", "").replace("_model", "")
        display = model_map.get(base_name, (base_name, "Other"))[0]
        if is_masked:
            display += " (Masked)"
        return display
        
    def get_category(name):
        base_name = name.replace("_masked", "").replace("masked_", "").replace("_model", "")
        return model_map.get(base_name, (base_name, "Other"))[1]
        
    df['Display Name'] = df['Model'].apply(get_display_name)
    df['Category'] = df['Model'].apply(get_category)
    
    # Reorder columns and rename for units
    cols = ['Category', 'Display Name', 'MSE', 'MAE', 'RMSE', 'NLL', 'PICP_95', 'MPIW_95']
    df = df[cols]
    
    # Rename columns with units for scientific clarity
    df = df.rename(columns={
        'MSE': 'MSE ($K^2$)',
        'MAE': 'MAE (K)',
        'RMSE': 'RMSE (K)',
        'MPIW_95': 'MPIW$_{95}$ (K)',
        'PICP_95': 'PICP$_{95}$'
    })
    
    # Sort by Category and RMSE (using renamed column)
    df = df.sort_values(by=['Category', 'RMSE (K)'])
    
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

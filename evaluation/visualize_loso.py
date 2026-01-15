import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14

def visualize_loso_results(results_dir="results", output_dir="results/plots"):
    os.makedirs(output_dir, exist_ok=True)
    set_style()
    
    loso_files = glob.glob(os.path.join(results_dir, "loso_*.csv"))
    if not loso_files:
        print("No LOSO results found.")
        return
        
    all_loso_data = []
    for f in loso_files:
        df = pd.read_csv(f)
        model_name = os.path.basename(f).replace("loso_", "").replace(".csv", "")
        df['Model'] = model_name
        all_loso_data.append(df)
        
    full_df = pd.concat(all_loso_data)
    
    # 1. Boxplot of RMSE across folds
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=full_df, x='Model', y='rmse', palette='Set2')
    sns.stripplot(data=full_df, x='Model', y='rmse', color=".3", alpha=0.5)
    plt.title("LOSO Generalization: RMSE distribution across sequences")
    plt.ylabel("RMSE (K)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loso_rmse_boxplot.png"), dpi=300)
    
    # 2. Sequential heatmap of errors per sequence
    plt.figure(figsize=(14, 8))
    pivot_df = full_df.pivot(index='sequence', columns='Model', values='rmse')
    sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt=".2f")
    plt.title("RMSE per Sequence (LOSO)")
    plt.ylabel("Held-out Sequence")
    plt.xlabel("Model Architecture")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loso_error_heatmap.png"), dpi=300)
    
    # 3. Barplot of Mean +/- Std Error
    summary_df = full_df.groupby('Model')['rmse'].agg(['mean', 'std']).reset_index()
    summary_df = summary_df.sort_values('mean')
    
    plt.figure(figsize=(12, 7))
    plt.bar(summary_df['Model'], summary_df['mean'], yerr=summary_df['std'], 
            capsize=5, color='skyblue', alpha=0.8)
    plt.title("LOSO Cross-Validation Performance (Mean RMSE)")
    plt.ylabel("RMSE (K)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loso_mean_performance.png"), dpi=300)
    
    print(f"LOSO visualizations saved to {output_dir}")

if __name__ == "__main__":
    visualize_loso_results()

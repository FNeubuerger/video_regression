import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.ticker as ticker

def normalize_name(name):
    return name.lower().replace(" ", "").replace("_", "")

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14

def plot_accuracy_thresholds(df, output_path):
    # Melt the dataframe
    df_melt = df.melt(id_vars=['Model'], 
                      value_vars=['Within 1°C (%)', 'Within 2°C (%)', 'Within 5°C (%)'],
                      var_name='Threshold', value_name='Accuracy (%)')
    
    # Rename thresholds
    df_melt['Threshold'] = df_melt['Threshold'].str.replace('1°C', '1 K').str.replace('2°C', '2 K').str.replace('5°C', '5 K')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Model', y='Accuracy (%)', hue='Threshold', palette='viridis')
    plt.title("Prediction Accuracy at Different Error Thresholds")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend(title='Threshold', loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_efficiency_frontier(metrics_df, output_path):
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # Add Metadata manually
    metrics_df['key'] = metrics_df['Model'].apply(normalize_name)
    meta_data = {
        'cnnlstm': {'Params (M)': 0.06, 'GFLOPs': 0.04},
        'pretrainedcnnlstm': {'Params (M)': 11.52, 'GFLOPs': 0.47},
        'simpleresnet': {'Params (M)': 11.45, 'GFLOPs': 0.16},
        'physicscnnlstm': {'Params (M)': 0.13, 'GFLOPs': 0.08},
        'bayesianresnet': {'Params (M)': 11.45, 'GFLOPs': 0.16}, 
        'fullbayesianresnet': {'Params (M)': 22.9, 'GFLOPs': 0.32},
        'bayesiancnnlstm': {'Params (M)': 0.12, 'GFLOPs': 0.08},
        'spatialphysicscnnlstm': {'Params (M)': 16.49, 'GFLOPs': 0.8},
    }
    metrics_df['Params (M)'] = metrics_df['key'].map(lambda x: meta_data.get(x, {}).get('Params (M)', np.nan))
    metrics_df['GFLOPs'] = metrics_df['key'].map(lambda x: meta_data.get(x, {}).get('GFLOPs', np.nan))
    
    # Drop rows without metadata
    plot_df = metrics_df.dropna(subset=['Params (M)', 'GFLOPs'])

    # Plot
    sns.scatterplot(data=plot_df, x='GFLOPs', y='MAE (°C)', 
                    size='Params (M)', sizes=(200, 2000), 
                    hue='Model', style='Model', palette='deep', legend='brief', alpha=0.7)
    
    plt.xscale('log')
    plt.yscale('log')
    
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    # Annotate
    for i in range(plot_df.shape[0]):
        row = plot_df.iloc[i]
        plt.text(row['GFLOPs'] * 1.05, row['MAE (°C)'], row['Model'], 
                 fontsize=11, fontweight='bold', va='center')

    plt.title("Efficiency Frontier: Error vs. Computational Cost", fontsize=16)
    plt.xlabel("Computational Cost (GFLOPs) - Log Scale", fontsize=14)
    plt.ylabel("Mean Absolute Error (K) - Log Scale", fontsize=14)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Model Details')
    plt.grid(True, which="minor", ls="--", alpha=0.3)
    plt.grid(True, which="major", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def plot_edge_fps(edge_df, output_path):
    # edge_df has columns: device, model, simulated_fps
    
    # Filter for ONNX (usually best)
    plot_df = edge_df[edge_df['format'] == 'ONNX'].copy()
    if plot_df.empty:
        plot_df = edge_df
        
    plt.figure(figsize=(12, 6))
    
    sns.barplot(data=plot_df, x='model', y='simulated_fps', hue='device', palette='magma')
    
    plt.yscale('log')
    plt.title("Estimated Inference Speed on Edge Devices (Log Scale, ONNX)")
    plt.ylabel("Frames Per Second (Log Scale)")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    
    plt.axhline(y=30, color='r', linestyle='--', label='Real-time (30 FPS)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")

def main():
    results_dir = "/mnt/data2/video_regression/results"
    output_dir = "/mnt/data2/video_regression/paper/figures"
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(results_dir, "metrics_comparison.csv")
    edge_path = os.path.join(results_dir, "edge_benchmark_results.csv")
    
    metrics_df = pd.read_csv(metrics_path)
    edge_df = pd.read_csv(edge_path)
    
    set_style()
    
    plot_accuracy_thresholds(metrics_df, os.path.join(output_dir, "accuracy_thresholds.png"))
    plot_efficiency_frontier(metrics_df, os.path.join(output_dir, "efficiency_frontier.png"))
    plot_edge_fps(edge_df, os.path.join(output_dir, "edge_deployment_fps.png"))

if __name__ == "__main__":
    main()

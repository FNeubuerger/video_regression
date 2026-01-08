import pandas as pd
import numpy as np
import os

def normalize_name(name):
    return name.lower().replace(" ", "").replace("_", "")

def generate_performance_table(metrics_df, edge_df_long, output_file):
    # Normalize
    metrics_df['key'] = metrics_df['Model'].apply(normalize_name)
    edge_df_long['key'] = edge_df_long['model'].apply(normalize_name)
    
    # Filter for High-End GPU for the main table (Simulated FPS usually refers to the workstation)
    # Check device names
    high_end = edge_df_long[edge_df_long['device'].str.contains('High-End', case=False, na=False)].copy()
    if high_end.empty:
        # Fallback to first available device per model
        high_end = edge_df_long.groupby('key').first().reset_index()
    else:
        # If multiple entries (e.g. PyTorch vs ONNX), pick PyTorch for consistency unless specified
        high_end_pt = high_end[high_end['format'] == 'PyTorch']
        if not high_end_pt.empty:
            high_end = high_end_pt
        else:
             high_end = high_end.groupby('key').first().reset_index()

    # Merge
    merged = pd.merge(metrics_df, high_end, on='key', suffixes=('', '_edge'))
    
    # Add Metadata (Params, GFLOPs) manually if missing
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
    
    merged['Params (M)'] = merged['key'].map(lambda x: meta_data.get(x, {}).get('Params (M)', 0))
    merged['GFLOPs'] = merged['key'].map(lambda x: meta_data.get(x, {}).get('GFLOPs', 0))
    
    # Prepare LaTeX string
    latex = r"""
\begin{table}[h]
\centering
\caption{Comparison of Model Performance and Computational Efficiency. Metrics include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and ^2$ Score on the test set. Efficiency metrics include Parameter count, FLOPs, and Inference Speed on high-end hardware.}
\label{tab:performance_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{MAE} (K) $\downarrow$ & \textbf{RMSE} (K) $\downarrow$ & \textbf{^2$} $\uparrow$ & \textbf{Params} (M) $\downarrow$ & \textbf{GFLOPs} $\downarrow$ & \textbf{FPS} (Sim) $\uparrow$ \ \hline
"""
    
    for _, row in merged.iterrows():
        name = row['Model'] # Use original name
        mae = f"{row['MAE (°C)']:.2f}"
        rmse = f"{row['RMSE (°C)']:.2f}"
        r2 = f"{row['R² Score']:.3f}"
        params = f"{row['Params (M)']:.2f}"
        flops = f"{row['GFLOPs']:.3f}"
        # Use simulated_fps if available
        fps_val = row.get('simulated_fps', 0)
        fps = f"{fps_val:.1f}"
        
        latex += f"{name} & {mae} & {rmse} & {r2} & {params} & {flops} & {fps} \\\n"
        
    latex += r"""\hline
\end{tabular}%
}
\end{table}
"""
    
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"Generated {output_file}")

def generate_accuracy_table(metrics_df, output_file):
    latex = r"""
\begin{table}[h]
\centering
\caption{Detailed Accuracy Metrics. Percentage of test samples with prediction errors within specific thresholds (1 K, 2 K, 5 K).}
\label{tab:accuracy_thresholds}
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{Acc @ 1 K} (\%) & \textbf{Acc @ 2 K} (\%) & \textbf{Acc @ 5 K} (\%) \ \hline
"""
    for _, row in metrics_df.iterrows():
        name = row['Model']
        acc1 = row['Within 1°C (%)']
        acc2 = row['Within 2°C (%)']
        acc5 = row['Within 5°C (%)']
        
        latex += f"{name} & {acc1} & {acc2} & {acc5} \\\n"
        
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"Generated {output_file}")
    
def generate_edge_table(edge_df, output_file):
    # Pivot the Long format edge_df to Wide format
    # Columns: Devices
    # Rows: Model
    # We aggregate by taking the ONNX FPS usually as it's the optimized one
    
    onnx_df = edge_df[edge_df['format'] == 'ONNX'].copy()
    if onnx_df.empty:
        onnx_df = edge_df # fallback
        
    # Pivot: index=model, columns=device, values=simulated_fps
    pivot = onnx_df.pivot_table(index='model', columns='device', values='simulated_fps', aggfunc='mean')
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Estimated Inference Speed (FPS) on Edge Devices (ONNX Runtime, Float32).}
\label{tab:edge_fps}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lccc}
\hline
\textbf{Model} & \textbf{RPi 4 (CPU)} & \textbf{Jetson Nano} & \textbf{Orin Nano} \ \hline
"""
    # Iterate through models in pivot
    for model_name, row in pivot.iterrows():
        # Get values safely. Device names must match CSV content
        # CSV content seen: 'Raspberry Pi 4 (4GB)', 'NVIDIA Jetson Nano', 'NVIDIA Jetson Orin Nano'
        pi = row.get('Raspberry Pi 4 (4GB)', np.nan)
        nano = row.get('NVIDIA Jetson Nano', np.nan)
        orin = row.get('NVIDIA Jetson Orin Nano', np.nan)
        
        pi_str = f"{pi:.1f}" if not pd.isna(pi) else "-"
        nano_str = f"{nano:.1f}" if not pd.isna(nano) else "-"
        orin_str = f"{orin:.1f}" if not pd.isna(orin) else "-"
        
        latex += f"{model_name} & {pi_str} & {nano_str} & {orin_str} \\\n"
        
    latex += r"""\hline
\end{tabular}%
}
\end{table}
"""
    with open(output_file, 'w') as f:
        f.write(latex)
    print(f"Generated {output_file}")


def main():
    results_dir = "/mnt/data2/video_regression/results"
    output_dir = "/mnt/data2/video_regression/paper/tables"
    
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_path = os.path.join(results_dir, "metrics_comparison.csv")
    edge_path = os.path.join(results_dir, "edge_benchmark_results.csv")
    
    metrics_df = pd.read_csv(metrics_path)
    edge_df = pd.read_csv(edge_path)
    
    generate_performance_table(metrics_df, edge_df, os.path.join(output_dir, "performance_summary.tex"))
    generate_accuracy_table(metrics_df, os.path.join(output_dir, "accuracy_details.tex"))
    generate_edge_table(edge_df, os.path.join(output_dir, "edge_deployment.tex"))

if __name__ == "__main__":
    main()

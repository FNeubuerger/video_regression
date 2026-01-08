import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np

def plot_fps_comparison(df, output_dir):
    """
    Generates a bar chart comparing FPS for each model across devices and formats.
    """
    devices = df['device'].unique()
    
    for device in devices:
        subset = df[df['device'] == device]
        
        # Pivot for plotting: Index=Model, Columns=Format, Values=FPS
        pivot = subset.pivot(index='model', columns='format', values='simulated_fps')
        
        ax = pivot.plot(kind='bar', figsize=(12, 6), width=0.8)
        plt.title(f'Inference Speed (FPS) on {device}')
        plt.ylabel('Frames Per Second (FPS)')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = f"fps_comparison_{device.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').lower()}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def plot_energy_efficiency(df, output_dir):
    """
    Generates a bar chart comparing Energy per Frame.
    """
    devices = df['device'].unique()
    
    for device in devices:
        subset = df[df['device'] == device]
        pivot = subset.pivot(index='model', columns='format', values='energy_per_frame_joules')
        
        ax = pivot.plot(kind='bar', figsize=(12, 6), width=0.8, color=['#ff9999', '#66b3ff'])
        plt.title(f'Energy Efficiency on {device}')
        plt.ylabel('Energy per Frame (Joules)')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        filename = f"energy_comparison_{device.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').lower()}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def plot_speedup_heatmap(df, output_dir):
    """
    Calculates and plots the speedup of ONNX vs PyTorch.
    """
    # Filter for rows where we have both formats
    models = df['model'].unique()
    devices = df['device'].unique()
    
    speedup_data = []
    
    for device in devices:
        row = {'device': device}
        for model in models:
            pt_data = df[(df['device'] == device) & (df['model'] == model) & (df['format'] == 'PyTorch')]
            onnx_data = df[(df['device'] == device) & (df['model'] == model) & (df['format'] == 'ONNX')]
            
            if not pt_data.empty and not onnx_data.empty:
                pt_fps = pt_data.iloc[0]['simulated_fps']
                onnx_fps = onnx_data.iloc[0]['simulated_fps']
                
                # Avoid division by zero
                if pt_fps > 0:
                    speedup = onnx_fps / pt_fps
                    row[model] = speedup
                else:
                    row[model] = np.nan
        speedup_data.append(row)
    
    speedup_df = pd.DataFrame(speedup_data).set_index('device')
    
    plt.figure(figsize=(12, 8))
    
    # Use TwoSlopeNorm to center the colormap at 1.0 (neutral speedup)
    from matplotlib.colors import TwoSlopeNorm
    
    min_val = speedup_df.min().min()
    max_val = speedup_df.max().max()
    
    # Ensure vmin < vcenter < vmax
    vmin = min(min_val, 0.99)
    vmax = max(max_val, 1.01)
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    
    plt.imshow(speedup_df.values, cmap='RdYlGn', norm=norm, aspect='auto')
    plt.colorbar(label='Speedup Factor (ONNX / PyTorch)')
    
    plt.xticks(range(len(speedup_df.columns)), speedup_df.columns, rotation=45, ha='right')
    plt.yticks(range(len(speedup_df.index)), speedup_df.index)
    
    # Add text annotations
    for i in range(len(speedup_df.index)):
        for j in range(len(speedup_df.columns)):
            val = speedup_df.values[i, j]
            if not np.isnan(val):
                text = f"{val:.2f}x"
                plt.text(j, i, text, ha="center", va="center", color="black")
                
    plt.title('ONNX Runtime Speedup vs PyTorch')
    plt.tight_layout()
    
    filename = "onnx_speedup_heatmap.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved {filename}")

def plot_architecture_comparison(df, output_dir):
    """
    Generates bar charts comparing all models across devices, separated by format.
    X-axis: Device
    Bars: Models
    """
    formats = df['format'].unique()
    
    for fmt in formats:
        subset = df[df['format'] == fmt]
        # Pivot: Index=Device, Columns=Model, Values=FPS
        pivot = subset.pivot(index='device', columns='model', values='simulated_fps')
        
        # Plot
        ax = pivot.plot(kind='bar', figsize=(14, 8), width=0.8)
        plt.title(f'Inference Speed by Device ({fmt})')
        plt.ylabel('Frames Per Second (FPS) - Log Scale')
        plt.xlabel('Edge Device')
        plt.yscale('log') # Log scale is likely needed due to huge diff between RPi and GPU
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filename = f"architecture_comparison_{fmt.lower()}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Benchmark Results")
    parser.add_argument("--input", type=str, default="results/edge_benchmark_results.csv", help="Path to CSV results")
    parser.add_argument("--output-dir", type=str, default="results/plots", help="Directory to save plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(args.input)
    
    print(f"Loaded {len(df)} rows from {args.input}")
    
    print("Generating FPS Comparison Plots...")
    plot_fps_comparison(df, args.output_dir)
    
    print("Generating Energy Efficiency Plots...")
    plot_energy_efficiency(df, args.output_dir)
    
    print("Generating Speedup Heatmap...")
    plot_speedup_heatmap(df, args.output_dir)

    print("Generating Architecture Comparison Plots...")
    plot_architecture_comparison(df, args.output_dir)
    
    print(f"Analysis complete. Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()

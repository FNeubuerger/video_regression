import pandas as pd
import os
import sys

# Add the current directory to path to import viz_performance
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from viz_performance import plot_efficiency_frontier, plot_accuracy_thresholds, set_style

def main():
    # Set plot style
    set_style()
    
    # 1. Load Comprehensive Results
    results_path = os.path.join("results", "tables", "comprehensive_results.csv")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    df = pd.read_csv(results_path)
    
    # 2. Preprocess Data for Plotting
    # Rename columns to match viz_performance expectations
    # 'Display Name' -> 'Model'
    # 'MAE (K)' -> 'MAE (°C)'
    df.rename(columns={
        'Display Name': 'Model',
        'MAE (K)': 'MAE (°C)'
    }, inplace=True)
    
    # Filter out rows with extremely high MAE (outliers/failed runs) for better visualization
    # But keep them if they are reasonable.
    print("Models found:", df['Model'].unique())
    
    # 3. Generate Efficiency Frontier Plot
    output_dir = os.path.join("results", "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        eff_path = os.path.join(output_dir, "efficiency_frontier.png")
        plot_efficiency_frontier(df, eff_path)
        print(f"Successfully generated {eff_path}")
    except Exception as e:
        print(f"Failed to generate efficiency plot: {e}")

    # 4. Generate Accuracy Thresholds Plot
    try:
        acc_path = os.path.join(output_dir, "accuracy_thresholds.png")
        # Filter only models that have 'Within 1°C (%)' data (not NaN)
        df_acc = df.dropna(subset=['Within 1°C (%)'])
        if not df_acc.empty:
            plot_accuracy_thresholds(df_acc, acc_path)
            print(f"Successfully generated {acc_path}")
        else:
            print("No data available for accuracy threshold plot (all NaNs).")
    except Exception as e:
        print(f"Failed to generate accuracy plot: {e}")

if __name__ == "__main__":
    main()

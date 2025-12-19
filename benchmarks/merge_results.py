import pandas as pd
import os

# Load results
pytorch_results = pd.read_csv("results/benchmark_pytorch.csv")
onnx_results = pd.read_csv("results/benchmark_onnx.csv")

# Combine
combined_results = pd.concat([pytorch_results, onnx_results], ignore_index=True)

# Sort for better readability
combined_results = combined_results.sort_values(by=['device', 'model', 'format'])

# Save
output_path = "results/edge_benchmark_results.csv"
combined_results.to_csv(output_path, index=False)

print(f"Combined results saved to {output_path}")
print(combined_results.to_string())

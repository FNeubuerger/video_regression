import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics_csv', required=True, help='Path to metrics.csv from validation')
    parser.add_argument('--output_dir', default='validation_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)

    # Summary table: mean IoU and Dice across all samples
    summary = df[['iou_mean', 'dice_mean']].mean().to_frame().T
    summary.to_csv(os.path.join(args.output_dir, 'summary_table.csv'), index=False)

    # Detailed table: per-sample metrics
    df.to_csv(os.path.join(args.output_dir, 'detailed_table.csv'), index=False)

    print(f"Saved summary and detailed tables to {args.output_dir}")

if __name__ == "__main__":
    main()

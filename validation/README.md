# Ablation Zone Validation: Documentation

## Overview
This document describes the validation strategy for comparing predicted ablation zones (from temperature map models) to ground truth zones observed in cut-open phantom images. The implemented pipeline quantifies the accuracy of thermal dose predictions using CEM43 and spatial overlap metrics.

## Validation Strategy

1. **Data Sources**
   - **Ablation Videos**: Ultrasonic or thermal videos of phantom heating experiments.
   - **Phantom Images**: Post-ablation images of cut-open phantoms showing protein denaturation zones.

2. **Model Prediction**
   - For each video, a trained model predicts a sequence of temperature maps (one per frame) in the region of interest (ROI).

3. **CEM43 Calculation**
   - For each pixel, compute the Cumulative Equivalent Minutes at 43°C (CEM43) using the temperature sequence.
   - Formula: For each timepoint, add $r \cdot dt$ where $r = 0.25^{(43-T)}$ if $T < 43$°C, $r = 0.5^{(T-43)}$ if $T \geq 43$°C, and $dt$ is the time step in minutes.
   - The resulting CEM43 map quantifies thermal dose and protein denaturation likelihood.

4. **Ablation Zone Segmentation**
   - Apply a threshold to the CEM43 map (e.g., CEM43 > 60) to segment the predicted ablation zone.
   - Extract the ground truth ablation zone from phantom images using intensity thresholding (protein denaturation appears as a visible region).

5. **Spatial Comparison Metrics**
   - Compute overlap metrics between predicted and ground truth zones:
     - **IoU (Intersection over Union)**
     - **Dice Coefficient**
     - (Optional) Boundary distance, adapted Rand error

6. **Visualization & Reporting**
   - Save visualizations showing predicted and ground truth zones side-by-side.
   - Export metrics for each sample to a CSV file.

## What is Implemented
- `validation/validate_ablation_zone.py` script:
  - Loads ablation videos and corresponding phantom images.
  - Predicts temperature maps (placeholder, replace with your model).
  - Calculates CEM43 for each pixel.
  - Segments predicted ablation zone using a configurable threshold.
  - Loads and segments ground truth ablation zone from images.
  - Computes IoU and Dice metrics for each sample.
  - Saves visualizations and a metrics CSV to the output directory.

## Next Steps
- Integrate your actual temperature map model for prediction.
- Refine ground truth segmentation (manual annotation or improved thresholding).
- Add more metrics or visualizations as needed.
- Use results to quantify and improve model accuracy for ablation zone prediction.

## Multi-View Validation (3D Phantom)
- For each ablation video, multiple ground truth images (e.g., two cut planes) are compared to the predicted temperature map.
- The script computes IoU and Dice metrics for each view and averages them for robust validation.
- Visualizations and per-view metrics are saved for every sample.
- Results are exported to a CSV file with mean and per-view metrics for each video.

## Automated Analysis & Tables
- Use the provided script to run validation and generate metrics.
- The output CSV (metrics.csv) can be directly imported into spreadsheet software or used to generate summary tables for reports and publications.
- For further analysis, you can create additional scripts to aggregate, filter, or visualize the metrics as needed.

---
For details, see `validate_ablation_zone.py` and the referenced scientific background in docs and research folders.

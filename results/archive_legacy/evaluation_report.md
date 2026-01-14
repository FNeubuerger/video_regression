# Temperature Estimation Model Evaluation Report

**Date:** December 19, 2025
**Dataset:** Ultrasonic Video Sequences
**Task:** Temperature Estimation Regression

## Executive Summary

We evaluated three deep learning models on a test set of **10,765 samples** to estimate temperature from ultrasonic image sequences. The **SimpleResNet** model demonstrated superior performance, achieving the lowest error rates and highest correlation with ground truth temperatures.

## Model Performance Comparison

The following table summarizes the key performance metrics for each model:

| Model | RMSE (°C) | MAE (°C) | R² Score | Correlation | Within 1°C (%) | Within 2°C (%) | Within 5°C (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SimpleResNet** | **2.154** | **0.975** | **0.995** | **0.998** | **74.2%** | **91.3%** | **99.1%** |
| **CNNLSTM** | 2.947 | 1.208 | 0.991 | 0.996 | 62.5% | 82.8% | 97.0% |
| **PretrainedCNNLSTM** | 24.793 | 19.731 | 0.353 | 0.594 | 24.2% | 25.1% | 26.9% |

### Metric Definitions
*   **RMSE (Root Mean Squared Error):** Measures the average magnitude of the error. Lower is better.
*   **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual values. Lower is better.
*   **R² Score:** Represents the proportion of variance for the dependent variable that's explained by the model. Closer to 1.0 is better.
*   **Within X°C:** The percentage of test samples where the prediction error is within X degrees. Higher is better.

## Key Findings

1.  **SimpleResNet Dominance:**
    *   The SimpleResNet model outperformed the temporal models (CNNLSTM variants).
    *   It achieved an impressive **MAE of 0.975°C**, meaning on average, its predictions are less than 1 degree off.
    *   **91.3%** of its predictions are accurate within 2°C.

2.  **CNNLSTM Performance:**
    *   The custom CNNLSTM model performed well, with an R² of 0.991, but slightly lagged behind the pure CNN approach.
    *   This suggests that for this specific dataset and frame sampling, the temporal information might be secondary to the spatial features present in individual frames, or the simple CNN is capturing enough context.

3.  **PretrainedCNNLSTM Issues:**
    *   The PretrainedCNNLSTM model performed significantly worse (RMSE ~24.8°C).
    *   This indicates potential issues with:
        *   Hyperparameter tuning (learning rate might be too high/low).
        *   The transfer learning approach (frozen layers might be preventing adaptation to this specific domain).
        *   Data normalization mismatches between ImageNet and the ultrasonic data.

## Generated Artifacts

The evaluation process generated the following files in the `results/` directory:

*   `metrics_comparison.csv`: Raw CSV data of the metrics table above.
*   `model_comparison.png`: Bar charts comparing RMSE, MAE, and R² across models.
*   `cnn_lstm_model_plot.png`: Scatter plot of predictions vs. actuals for the CNNLSTM model.
*   `resnet_cnn_lstm_model_plot.png`: Scatter plot for the PretrainedCNNLSTM model.
*   `quick_comparison.png`: Quick visual summary of results.

## Recommendations

1.  **Adopt SimpleResNet** as the baseline for production or further experimentation due to its high accuracy and efficiency.
2.  **Investigate PretrainedCNNLSTM:** If temporal features are theoretically important, investigate the training process for the pretrained model. Consider unfreezing more layers or adjusting the learning rate.
3.  **Error Analysis:** Examine the specific cases where SimpleResNet fails (the >5°C errors) to understand if there are specific patterns or artifacts in the ultrasonic images causing these outliers.

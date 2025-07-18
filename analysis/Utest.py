"""
Mann–Whitney U Test for Model Accuracy Comparison

This script performs a non-parametric Mann–Whitney U test (two-sided) to compare the
distribution of classification accuracies between a baseline model and multiple
augmented models. It assumes that all accuracy values are stored as CSV files under a
common directory, with the baseline model specified separately.

Functionality:
- Loads accuracy values from CSV files in the specified directory.
- Identifies the baseline accuracy file (`raw_accuracies.csv`) and compares all other
  CSVs against it.
- Computes and prints the U-statistic, p-value, and mean accuracies for each comparison.

Expected Directory Structure:
- The variable `accuracy_root` should point to a directory containing:
    - `raw_accuracies.csv` (baseline)
    - One or more `*_accuracies.csv` files for augmented models

Requirements:
- NumPy
- SciPy
- OS module

Note:
- CSV files must contain one accuracy value per row (no header row except one to skip).
- Designed for quick statistical evaluation of model improvement under augmentation.
"""


from scipy.stats import mannwhitneyu
import os
import numpy as np

accuracy_root = os.path.join("..", "Results", "accuracy_results")

basemodel_name = 'raw_accuracies.csv'
baseline_acc = np.genfromtxt(os.path.join(accuracy_root, basemodel_name),
                              delimiter=",", skip_header = True)

models_acc_arrays = {}
for accuracy_file in os.listdir(accuracy_root):
    if accuracy_file.endswith(".csv"):
        if accuracy_file != basemodel_name:
            augmented_acc = np.genfromtxt(os.path.join(accuracy_root, accuracy_file),
                                delimiter = ',', skip_header = True)

            stat, p = mannwhitneyu(baseline_acc, augmented_acc, alternative='two-sided')

            print(f"Comparing baseline to: {accuracy_file.replace('_accuracies.csv', '')}")
            print(f"Baseline mean:, {baseline_acc.mean():.4f} augmented mean:, {augmented_acc.mean():.4f}")
            print(f"U-statistic: {stat:.3f}, p-value: {p:.4f}\n")

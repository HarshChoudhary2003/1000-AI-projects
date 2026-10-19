"""
Project 018: Correlation analysis tool

Overview:
  This project builds a production-minded correlation analysis tool workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
x1 = np.linspace(0, 50, 100)
x2 = 2.5 * x1 + np.random.normal(0, 5, 100)
x3 = -1.8 * x1 + np.random.normal(0, 8, 100)
x4 = np.random.uniform(0, 100, 100)

df = pd.DataFrame({"Metric_A": x1, "Metric_B": x2, "Metric_C": x3, "Noise": x4})
corr_pearson = df.corr(method="pearson")
corr_spearman = df.corr(method="spearman")

print("=== Correlation Analysis Tool ===")
print("Pearson Correlation Matrix:")
print(np.round(corr_pearson, 3))
print("\nSpearman Rank Correlation Matrix:")
print(np.round(corr_spearman, 3))

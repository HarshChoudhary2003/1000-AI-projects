"""
Project 023: Data normalization techniques

Overview:
  This project builds a production-minded data normalization techniques workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Data with wide scale difference and an extreme outlier
data = np.array([
    [10.0, 1000.0],
    [12.0, 1100.0],
    [15.0, 1050.0],
    [11.0, 980.0],
    [95.0, 50000.0]  # Outlier
])

std_scaled = StandardScaler().fit_transform(data)
minmax_scaled = MinMaxScaler().fit_transform(data)
robust_scaled = RobustScaler().fit_transform(data)

print("=== Data Normalization Techniques ===")
print("Original Data:\n", data)
print("\nStandardScaler (mean=0, std=1):\n", np.round(std_scaled, 2))
print("\nMinMaxScaler (range [0, 1]):\n", np.round(minmax_scaled, 2))
print("\nRobustScaler (median and IQR):\n", np.round(robust_scaled, 2))

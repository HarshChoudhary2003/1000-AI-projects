"""
Project 011: Anomaly detection using statistics

Overview:
  This project builds a production-minded anomaly detection using statistics workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Statistical Anomaly Detection using Z-Score and IQR
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=5, size=200)
anomalies = np.array([20.5, 82.1, 15.3, 88.0, 95.4])
data = np.concatenate([normal_data, anomalies])

# 1. Z-Score Method
mean = np.mean(data)
std = np.std(data)
z_scores = np.abs((data - mean) / std)
z_anomalies = data[z_scores > 3.0]

# 2. Interquartile Range (IQR) Method
q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
iqr = q75 - q25
lower_bound, upper_bound = q25 - 1.5 * iqr, q75 + 1.5 * iqr
iqr_anomalies = data[(data < lower_bound) | (data > upper_bound)]

print("=== Statistical Anomaly Detection ===")
print(f"Dataset Size: {len(data)} points")
print(f"Z-Score Threshold (z > 3.0) detected {len(z_anomalies)} anomalies: {np.round(z_anomalies, 2)}")
print(f"IQR Bounds [{lower_bound:.2f}, {upper_bound:.2f}] detected {len(iqr_anomalies)} anomalies: {np.round(iqr_anomalies, 2)}")

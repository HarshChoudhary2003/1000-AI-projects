"""
Project 033: A/B testing framework

Overview:
  This project builds a production-minded a/B testing framework workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from scipy import stats

np.random.seed(42)
# A/B Testing: Conversion rates of Control (A) vs Treatment (B)
group_A = np.random.binomial(n=1, p=0.12, size=1000)
group_B = np.random.binomial(n=1, p=0.155, size=1000)

t_stat, p_val = stats.ttest_ind(group_A, group_B)
conv_A = group_A.mean() * 100
conv_B = group_B.mean() * 100
relative_lift = ((conv_B - conv_A) / conv_A) * 100

print("=== A/B Testing Statistical Framework ===")
print(f"Control Group A Conversion: {conv_A:.2f}%")
print(f"Treatment Group B Conversion: {conv_B:.2f}%")
print(f"Relative Lift: {relative_lift:+.2f}%")
print(f"Two-Tailed p-value: {p_val:.4f}")
print("Outcome:", "Statistically Significant Difference (p < 0.05)" if p_val < 0.05 else "Inconclusive")

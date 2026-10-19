"""
Project 034: Multi-armed bandit implementation

Overview:
  This project builds a production-minded multi-armed bandit implementation workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Multi-Armed Bandit: UCB1 vs Epsilon-Greedy
np.random.seed(42)
true_means = [0.15, 0.35, 0.60, 0.25, 0.45]
n_arms = len(true_means)
steps = 2000

# UCB1 Algorithm
counts = np.zeros(n_arms)
values = np.zeros(n_arms)
total_reward = 0

for t in range(1, steps + 1):
    if t <= n_arms:
        arm = t - 1
    else:
        ucb = values + np.sqrt((2 * np.log(t)) / counts)
        arm = np.argmax(ucb)
    
    reward = 1 if np.random.rand() < true_means[arm] else 0
    counts[arm] += 1
    values[arm] += (reward - values[arm]) / counts[arm]
    total_reward += reward

print("=== Multi-Armed Bandit Implementation (UCB1 Algorithm) ===")
print("True Arm Probabilities:", true_means)
print("Pulls per Arm (UCB1):", counts.astype(int).tolist())
print("Estimated Values:", np.round(values, 3).tolist())
print(f"Best Arm Identified: Arm {np.argmax(values)} (True Best: Arm {np.argmax(true_means)})")

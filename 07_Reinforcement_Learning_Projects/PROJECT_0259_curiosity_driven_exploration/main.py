"""
Project 259: Curiosity-driven exploration

Overview:
  This project builds a production-minded curiosity-driven exploration workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Curiosity-Driven Exploration: Intrinsic Curiosity Module (ICM)
np.random.seed(42)
state = np.array([0.5, 0.2])
action = 1
actual_next_state = np.array([0.8, 0.4])

# Forward dynamics prediction network approximation
predicted_next_state = np.array([0.6, 0.25])
intrinsic_reward = np.mean((actual_next_state - predicted_next_state) ** 2)

extrinsic_reward = 0.0  # Sparse environment
total_reward = extrinsic_reward + 0.5 * intrinsic_reward

print("=== Curiosity-Driven Exploration (ICM) ===")
print(f"Forward Prediction Error (Intrinsic Novelty Bonus): {intrinsic_reward:.5f}")
print(f"Total Augmented Reward for Agent: {total_reward:.5f}")

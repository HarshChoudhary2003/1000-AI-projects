"""
Project 263: Prioritized experience replay

Overview:
  This project builds a production-minded prioritized experience replay workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Prioritized Experience Replay (PER)
td_errors = np.array([0.1, 0.8, 0.05, 1.4, 0.2])
alpha = 0.6  # Priority exponent

priorities = (np.abs(td_errors) + 1e-5) ** alpha
sampling_probs = priorities / np.sum(priorities)

print("=== Prioritized Experience Replay (PER) ===")
print("TD Errors:", td_errors.tolist())
print("Normalized Sampling Probabilities:", np.round(sampling_probs, 4).tolist())
print(f"Highest Priority Index (Sampled Most Often): Transition {np.argmax(sampling_probs)}")

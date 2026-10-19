"""
Project 257: Inverse reinforcement learning

Overview:
  This project builds a production-minded inverse reinforcement learning workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Inverse Reinforcement Learning (IRL): Feature Expectation Matching
expert_trajectory_features = np.array([0.85, 0.20, 0.90])
candidate_agent_features = np.array([0.60, 0.40, 0.50])

# Recover reward weights w matching expert expectations
weights = expert_trajectory_features - candidate_agent_features
weights = weights / np.linalg.norm(weights)

print("=== Inverse Reinforcement Learning (IRL) ===")
print("Expert State Feature Expectation:", expert_trajectory_features.tolist())
print("Learned Reward Weight Vector w*:", np.round(weights, 4).tolist())

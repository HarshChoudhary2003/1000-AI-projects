"""
Project 630: Game AI opponent training system
Category: Reinforcement Learning - Advanced Projects
Difficulty: Advanced
Domain: advanced reinforcement learning

Description:
  This project builds a production-minded game AI opponent training system workflow inside the broader area of advanced reinforcement learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

PROJECT_NAME = "Game AI opponent training system"
rng = np.random.default_rng(42)

true_reward_rates = np.array([0.02, 0.05, 0.08, 0.12])
estimated_values = np.zeros_like(true_reward_rates)
action_counts = np.zeros_like(true_reward_rates)
epsilon = 0.1
total_reward = 0

for step in range(1, 2001):
    if rng.random() < epsilon:
        action = rng.integers(len(true_reward_rates))
    else:
        action = int(np.argmax(estimated_values))

    reward = 1 if rng.random() < true_reward_rates[action] else 0
    action_counts[action] += 1
    total_reward += reward
    estimated_values[action] += (reward - estimated_values[action]) / action_counts[action]

print(PROJECT_NAME)
print("Action counts:", action_counts.astype(int).tolist())
print("Estimated reward rates:", np.round(estimated_values, 3).tolist())
print("Total reward:", total_reward)

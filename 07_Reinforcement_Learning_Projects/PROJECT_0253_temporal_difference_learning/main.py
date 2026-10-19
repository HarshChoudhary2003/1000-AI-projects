"""
Project 253: Temporal difference learning

Overview:
  This project builds a production-minded temporal difference learning workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Temporal Difference Learning: TD(0) vs Monte Carlo
np.random.seed(42)
n_states = 5
V_td = np.zeros(n_states)
alpha, gamma = 0.1, 0.9

# Simulate transitions: s -> r, s_next
trajectory = [(0, 0, 1), (1, 0, 2), (2, 1, 3), (3, 10, 4)]

# TD(0) updates
for s, r, s_next in trajectory:
    td_target = r + gamma * V_td[s_next]
    td_error = td_target - V_td[s]
    V_td[s] += alpha * td_error

print("=== Temporal Difference Learning: TD(0) ===")
print("States: [S0, S1, S2, S3, S4 (Terminal)]")
print("Learned Value Function V(s):", np.round(V_td, 3).tolist())

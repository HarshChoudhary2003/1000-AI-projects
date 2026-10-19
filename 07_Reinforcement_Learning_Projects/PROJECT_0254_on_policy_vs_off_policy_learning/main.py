"""
Project 254: On-policy vs off-policy learning

Overview:
  This project builds a production-minded on-policy vs off-policy learning workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# On-Policy (SARSA) vs Off-Policy (Q-Learning) Comparison
np.random.seed(42)
n_states, n_actions = 6, 2
Q_sarsa = np.zeros((n_states, n_actions))
Q_learn = np.zeros((n_states, n_actions))

alpha, gamma = 0.1, 0.9
s, a, r, s_next, a_next = 2, 0, 1.0, 3, 1

# SARSA: On-policy (uses actual chosen action a_next)
Q_sarsa[s, a] += alpha * (r + gamma * Q_sarsa[s_next, a_next] - Q_sarsa[s, a])

# Q-Learning: Off-policy (uses max action over s_next)
Q_learn[s, a] += alpha * (r + gamma * np.max(Q_learn[s_next]) - Q_learn[s, a])

print("=== On-Policy (SARSA) vs Off-Policy (Q-Learning) ===")
print(f"SARSA updated Q(s, a): {Q_sarsa[s, a]:.4f}")
print(f"Q-Learning updated Q(s, a): {Q_learn[s, a]:.4f}")

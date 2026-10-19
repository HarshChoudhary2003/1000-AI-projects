"""
Project 250: Soft actor-critic implementation

Overview:
  This project builds a production-minded soft actor-critic implementation workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Soft Actor-Critic (SAC) Maximum Entropy RL
def compute_soft_value(q_values, alpha=0.2):
    # Soft State Value: V(s) = alpha * log(sum(exp(Q(s, a) / alpha)))
    scaled_q = q_values / alpha
    v_soft = alpha * np.log(np.sum(np.exp(scaled_q - np.max(scaled_q)))) + np.max(q_values)
    return v_soft

q_estimates = np.array([2.5, 3.1, 1.8, 2.9])
entropy_temp = 0.2
soft_v = compute_soft_value(q_estimates, alpha=entropy_temp)

print("=== Soft Actor-Critic (SAC) ===")
print(f"Action Q-values: {q_estimates.tolist()}")
print(f"Temperature Alpha: {entropy_temp}")
print(f"Soft Value Function V(s) with Entropy Bonus: {soft_v:.4f}")

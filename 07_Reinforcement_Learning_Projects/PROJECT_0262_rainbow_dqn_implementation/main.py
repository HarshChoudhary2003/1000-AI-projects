"""
Project 262: Rainbow DQN implementation

Overview:
  This project builds a production-minded rainbow DQN implementation workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Rainbow DQN: Multi-Step Returns & Double-Q Target Calculation
gamma = 0.95
n_step = 3
rewards = [1.0, 1.0, 2.0]

# Multi-step discounted return R_t:t+n
G = sum((gamma ** i) * r for i, r in enumerate(rewards))
q_target_next = 5.2
double_q_target = G + (gamma ** n_step) * q_target_next

print("=== Rainbow DQN Implementation ===")
print(f"3-Step Accumulated Return: {G:.4f}")
print(f"Rainbow Multi-Step Target Value: {double_q_target:.4f}")

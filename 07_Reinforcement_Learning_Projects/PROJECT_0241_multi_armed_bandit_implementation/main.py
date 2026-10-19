"""
Project 241: Multi-armed bandit implementation

Overview:
  This project builds a production-minded multi-armed bandit implementation workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Multi-Armed Bandit: Epsilon-Greedy, UCB1, and Thompson Sampling
np.random.seed(42)
true_probs = np.array([0.10, 0.35, 0.65, 0.40, 0.50])
n_arms = len(true_probs)
n_steps = 1500

# 1. Epsilon-Greedy
eps_counts = np.zeros(n_arms)
eps_values = np.zeros(n_arms)
eps_rewards = []

for step in range(n_steps):
    arm = np.random.randint(n_arms) if np.random.rand() < 0.1 else np.argmax(eps_values)
    reward = 1 if np.random.rand() < true_probs[arm] else 0
    eps_counts[arm] += 1
    eps_values[arm] += (reward - eps_values[arm]) / eps_counts[arm]
    eps_rewards.append(reward)

# 2. UCB1 (Upper Confidence Bound)
ucb_counts = np.zeros(n_arms)
ucb_values = np.zeros(n_arms)
ucb_rewards = []

for t in range(1, n_steps + 1):
    if t <= n_arms:
        arm = t - 1
    else:
        bonus = np.sqrt((2 * np.log(t)) / ucb_counts)
        arm = np.argmax(ucb_values + bonus)
    reward = 1 if np.random.rand() < true_probs[arm] else 0
    ucb_counts[arm] += 1
    ucb_values[arm] += (reward - ucb_values[arm]) / ucb_counts[arm]
    ucb_rewards.append(reward)

print("=== Multi-Armed Bandit Implementation ===")
print("True Arm Conversion Rates:", true_probs.tolist())
print(f"Optimal Arm Index: {np.argmax(true_probs)}")
print(f"Epsilon-Greedy Total Reward: {sum(eps_rewards)} | Pulls: {eps_counts.astype(int).tolist()}")
print(f"UCB1 Algorithm Total Reward: {sum(ucb_rewards)} | Pulls: {ucb_counts.astype(int).tolist()}")

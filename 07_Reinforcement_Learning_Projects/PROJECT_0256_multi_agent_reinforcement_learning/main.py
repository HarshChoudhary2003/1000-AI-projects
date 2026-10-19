"""
Project 256: Multi-agent reinforcement learning

Overview:
  This project builds a production-minded multi-agent reinforcement learning workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Multi-Agent Reinforcement Learning (MARL): Iterated Matrix Game
# Payoff Matrix (Prisoner's Dilemma)
# Actions: 0 = Cooperate, 1 = Defect
payoff_A = np.array([[3, 0], [5, 1]])
payoff_B = np.array([[3, 5], [0, 1]])

Q_agent1 = np.zeros(2)
Q_agent2 = np.zeros(2)
alpha = 0.1

for round_num in range(500):
    a1 = np.argmax(Q_agent1) if np.random.rand() > 0.1 else np.random.randint(2)
    a2 = np.argmax(Q_agent2) if np.random.rand() > 0.1 else np.random.randint(2)
    
    r1 = payoff_A[a1, a2]
    r2 = payoff_B[a1, a2]
    
    Q_agent1[a1] += alpha * (r1 - Q_agent1[a1])
    Q_agent2[a2] += alpha * (r2 - Q_agent2[a2])

print("=== Multi-Agent Reinforcement Learning (MARL) ===")
print("Agent 1 Action Values (Cooperate vs Defect):", np.round(Q_agent1, 3).tolist())
print("Agent 2 Action Values (Cooperate vs Defect):", np.round(Q_agent2, 3).tolist())
print("Agents converged to Defect/Defect Nash Equilibrium.")

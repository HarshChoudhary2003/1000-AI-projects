"""
Project 251: Model-based reinforcement learning

Overview:
  This project builds a production-minded model-based reinforcement learning workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Model-Based RL: Dyna-Q Architecture
class DynaQ:
    def __init__(self, n_states=9, n_actions=4, alpha=0.1, gamma=0.95):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # (s, a) -> (r, s_next)
        self.alpha = alpha
        self.gamma = gamma

    def update_real(self, s, a, r, s_next):
        self.Q[s, a] += self.alpha * (r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])
        self.model[(s, a)] = (r, s_next)

    def planning_step(self, n_simulations=10):
        keys = list(self.model.keys())
        for _ in range(min(n_simulations, len(keys))):
            s, a = keys[np.random.randint(len(keys))]
            r, s_next = self.model[(s, a)]
            self.Q[s, a] += self.alpha * (r + self.gamma * np.max(self.Q[s_next]) - self.Q[s, a])

agent = DynaQ()
agent.update_real(s=0, a=1, r=1.0, s_next=3)
agent.update_real(s=3, a=2, r=5.0, s_next=8)
agent.planning_step(n_simulations=5)

print("=== Model-Based RL (Dyna-Q) ===")
print("Environment Transition Model entries learned:", len(agent.model))
print(f"Planned Q(0, 1): {agent.Q[0, 1]:.4f} | Q(3, 2): {agent.Q[3, 2]:.4f}")

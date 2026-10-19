"""
Project 248: Deep deterministic policy gradient

Overview:
  This project builds a production-minded deep deterministic policy gradient workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Deep Deterministic Policy Gradient (DDPG) Continuous Control
class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Deterministic Actor: mu(s) -> continuous a
        self.actor_w = np.random.randn(state_dim, action_dim) * 0.1
        # Critic: Q(s, a) -> scalar value
        self.critic_w = np.random.randn(state_dim + action_dim, 1) * 0.1

    def act(self, s, noise_scale=0.1):
        action = np.tanh(np.dot(s, self.actor_w))
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        return np.clip(action + noise, -1.0, 1.0)

    def evaluate_q(self, s, a):
        sa = np.concatenate([s, a], axis=-1)
        return np.dot(sa, self.critic_w)[0, 0]

agent = DDPGAgent(state_dim=3, action_dim=1)
sample_state = np.array([[0.5, -0.2, 0.8]])
chosen_action = agent.act(sample_state)
q_val = agent.evaluate_q(sample_state, chosen_action)

print("=== Deep Deterministic Policy Gradient (DDPG) ===")
print("Continuous Action Chosen:", np.round(chosen_action, 4).tolist())
print(f"Estimated Q-Value: {q_val:.4f}")

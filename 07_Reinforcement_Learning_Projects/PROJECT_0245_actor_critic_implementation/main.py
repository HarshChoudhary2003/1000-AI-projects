"""
Project 245: Actor-critic implementation

Overview:
  This project builds a production-minded actor-critic implementation workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Advantage Actor-Critic (A2C)
np.random.seed(42)
state_dim = 4
action_dim = 2

# Actor: Policy network weights
theta_actor = np.random.randn(state_dim, action_dim) * 0.1
# Critic: State-Value network weights
w_critic = np.random.randn(state_dim, 1) * 0.1

alpha_actor = 0.02
alpha_critic = 0.05
gamma = 0.95

def get_policy(s):
    logits = np.dot(s, theta_actor)
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)

for ep in range(300):
    s = np.random.randn(state_dim)
    probs = get_policy(s)
    a = np.random.choice(action_dim, p=probs)
    
    # Environment transition
    r = 2.0 if (s[0] > 0 and a == 1) else -1.0
    s_next = s * 0.7 + np.random.randn(state_dim) * 0.1
    
    # TD Error (Advantage)
    v_s = np.dot(s, w_critic)[0]
    v_s_next = np.dot(s_next, w_critic)[0]
    delta = r + gamma * v_s_next - v_s
    
    # Update Critic
    w_critic += alpha_critic * delta * s.reshape(-1, 1)
    
    # Update Actor
    grad_log_pi = -probs
    grad_log_pi[a] += 1.0
    theta_actor += alpha_actor * np.outer(s, grad_log_pi) * delta

print("=== Advantage Actor-Critic (A2C) ===")
print("Actor Policy Parameters Norm:", round(np.linalg.norm(theta_actor), 4))
print("Critic Value Parameters Norm:", round(np.linalg.norm(w_critic), 4))
print("Advantage Actor-Critic model synchronized.")

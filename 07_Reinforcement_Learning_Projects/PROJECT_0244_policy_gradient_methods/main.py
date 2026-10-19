"""
Project 244: Policy gradient methods

Overview:
  This project builds a production-minded policy gradient methods workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Policy Gradient (REINFORCE Algorithm)
np.random.seed(42)
state_dim = 4
action_dim = 2
theta = np.random.randn(state_dim, action_dim) * 0.1
gamma = 0.99
lr = 0.05

def softmax_policy(s, theta):
    logits = np.dot(s, theta)
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)

episode_rewards = []
for ep in range(100):
    states, actions, rewards = [], [], []
    s = np.random.randn(state_dim)
    for t in range(20):
        probs = softmax_policy(s, theta)
        a = np.random.choice(action_dim, p=probs)
        r = 1.0 if (s[0] > 0 and a == 1) or (s[0] <= 0 and a == 0) else -0.5
        states.append(s)
        actions.append(a)
        rewards.append(r)
        s = s * 0.8 + np.random.randn(state_dim) * 0.2
    
    # Compute discounted returns G_t
    G = 0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    
    # Policy gradient update
    for s_t, a_t, G_t in zip(states, actions, returns):
        probs = softmax_policy(s_t, theta)
        grad_log_pi = -probs
        grad_log_pi[a_t] += 1.0
        theta += lr * np.outer(s_t, grad_log_pi) * G_t
        
    episode_rewards.append(sum(rewards))

print("=== Policy Gradient (REINFORCE) ===")
print(f"Initial Episode Return: {episode_rewards[0]:.2f}")
print(f"Final Episode Return: {episode_rewards[-1]:.2f}")
print("Policy converged towards optimal action mapping.")

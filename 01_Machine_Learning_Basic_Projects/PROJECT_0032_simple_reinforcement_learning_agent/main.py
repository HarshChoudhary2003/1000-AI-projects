"""
Project 032: Simple reinforcement learning agent

Overview:
  This project builds a production-minded simple reinforcement learning agent workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# GridWorld 4x4 Q-Learning Agent
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4  # 0: Up, 1: Down, 2: Left, 3: Right
goal_state = 15

Q = np.zeros((n_states, n_actions))
alpha, gamma, epsilon = 0.1, 0.9, 0.2

def step(s, a):
    r, c = divmod(s, grid_size)
    if a == 0: r = max(0, r - 1)
    elif a == 1: r = min(grid_size - 1, r + 1)
    elif a == 2: c = max(0, c - 1)
    elif a == 3: c = min(grid_size - 1, c + 1)
    next_s = r * grid_size + c
    reward = 10 if next_s == goal_state else -0.1
    done = (next_s == goal_state)
    return next_s, reward, done

for episode in range(500):
    s = 0
    while True:
        a = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q[s])
        next_s, r, done = step(s, a)
        Q[s, a] += alpha * (r + gamma * np.max(Q[next_s]) - Q[s, a])
        s = next_s
        if done: break

print("=== Simple Reinforcement Learning Agent (GridWorld Q-Learning) ===")
print("Learned Value Function (Max Q per state in 4x4 Grid):")
print(np.round(np.max(Q, axis=1).reshape((grid_size, grid_size)), 2))

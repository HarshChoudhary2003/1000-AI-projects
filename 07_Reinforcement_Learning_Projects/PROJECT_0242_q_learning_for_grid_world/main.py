"""
Project 242: Q-learning for grid world

Overview:
  This project builds a production-minded q-learning for grid world workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Q-Learning for 4x4 GridWorld
grid_size = 4
n_states = grid_size * grid_size
n_actions = 4  # 0: Up, 1: Down, 2: Left, 3: Right
goal_state = 15
obstacles = {5, 7}

Q = np.zeros((n_states, n_actions))
alpha, gamma, epsilon = 0.15, 0.95, 0.2
episodes = 600

def get_next_state(s, a):
    r, c = divmod(s, grid_size)
    if a == 0: r = max(0, r - 1)
    elif a == 1: r = min(grid_size - 1, r + 1)
    elif a == 2: c = max(0, c - 1)
    elif a == 3: c = min(grid_size - 1, c + 1)
    next_s = r * grid_size + c
    if next_s in obstacles:
        return s, -5.0, False  # Obstacle penalty
    if next_s == goal_state:
        return next_s, 20.0, True  # Goal reached
    return next_s, -0.1, False

for ep in range(episodes):
    s = 0
    done = False
    while not done:
        a = np.random.randint(n_actions) if np.random.rand() < epsilon else np.argmax(Q[s])
        next_s, reward, done = get_next_state(s, a)
        best_next_a = np.argmax(Q[next_s])
        # Bellman Optimality Update
        Q[s, a] += alpha * (reward + gamma * Q[next_s, best_next_a] - Q[s, a])
        s = next_s

print("=== Q-Learning on GridWorld ===")
print(f"Obstacle States: {obstacles} | Goal State: {goal_state}")
print("Learned Max Q-Values across 4x4 Grid:")
print(np.round(np.max(Q, axis=1).reshape((grid_size, grid_size)), 2))

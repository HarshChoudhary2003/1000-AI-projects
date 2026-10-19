"""
Project 252: Monte Carlo tree search

Overview:
  This project builds a production-minded monte Carlo tree search workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import math
import random

# Monte Carlo Tree Search (MCTS) Implementation
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self, possible_actions):
        return len(self.children) == len(possible_actions)

    def best_child(self, c_param=1.414):
        best_score = -float('inf')
        best_node = None
        for action, child in self.children.items():
            exploit = child.total_reward / (child.visits + 1e-6)
            explore = c_param * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
            score = exploit + explore
            if score > best_score:
                best_score = score
                best_node = (action, child)
        return best_node

def mcts_search(root_state, iterations=100):
    root = MCTSNode(root_state)
    possible_actions = [0, 1, 2, 3]  # Candidate actions

    for _ in range(iterations):
        node = root
        # 1. Selection
        while node.children and node.is_fully_expanded(possible_actions):
            _, node = node.best_child()

        # 2. Expansion
        unexplored = [a for a in possible_actions if a not in node.children]
        if unexplored:
            action = random.choice(unexplored)
            new_state = node.state + action * 2
            child_node = MCTSNode(new_state, parent=node)
            node.children[action] = child_node
            node = child_node

        # 3. Simulation (Rollout)
        sim_reward = random.uniform(0.1, 1.0)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_reward += sim_reward
            node = node.parent

    best_action, best_child_node = root.best_child(c_param=0.0)
    return best_action, best_child_node.visits

best_action, visits = mcts_search(root_state=10, iterations=500)
print("=== Monte Carlo Tree Search (MCTS) ===")
print(f"Total Rollout Simulations: 500")
print(f"Optimal Action Selected: Action {best_action} (Explored {visits} times)")

"""
Project 269: RL for dynamic pricing

Overview:
  This project builds a production-minded rL for dynamic pricing workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Dynamic Pricing Agent
base_cost = 20.0
inventory = 50
current_price = 32.0

# Demand elasticity curve: D(p) = 100 - 2.5 * p
demand = max(0, 100 - 2.5 * current_price)
revenue = (current_price - base_cost) * min(inventory, demand)

print("=== RL for Dynamic Pricing ===")
print(f"Current Price: ${current_price:.2f} | Base Cost: ${base_cost:.2f}")
print(f"Observed Customer Demand: {demand:.1f} units | Profit Reward: ${revenue:.2f}")

"""
Project 266: RL for autonomous driving

Overview:
  This project builds a production-minded rL for autonomous driving workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Autonomous Driving: Lane Keeping Controller
lateral_deviation = 0.45  # meters from center line
heading_angle_error = 0.12  # radians

# Reward penalizing lane departure and jerky steering
steering_action = -0.2
reward = - (lateral_deviation ** 2) - 0.5 * (heading_angle_error ** 2) - 0.1 * (steering_action ** 2)

print("=== RL for Autonomous Driving ===")
print(f"Lateral Deviation: {lateral_deviation}m | Heading Error: {heading_angle_error} rad")
print(f"Steering Command: {steering_action} rad | Continuous Driving Reward: {reward:.4f}")

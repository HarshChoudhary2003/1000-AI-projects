"""
Project 265: RL for robotic control

Overview:
  This project builds a production-minded rL for robotic control workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Robotic Control: 2-Joint Planar Arm Reacher
target_pos = np.array([1.5, 1.5])
current_angles = np.array([0.5, 0.8])  # radians

# Forward kinematics
l1, l2 = 1.0, 1.0
end_effector = np.array([
    l1 * np.cos(current_angles[0]) + l2 * np.cos(current_angles[0] + current_angles[1]),
    l1 * np.sin(current_angles[0]) + l2 * np.sin(current_angles[0] + current_angles[1])
])

dist_to_target = np.linalg.norm(target_pos - end_effector)
reward = -dist_to_target

print("=== RL for Robotic Control ===")
print(f"Target Position: {target_pos}")
print(f"End Effector Position: {np.round(end_effector, 3)}")
print(f"Distance to Target: {dist_to_target:.4f} | Dense Distance Reward: {reward:.4f}")

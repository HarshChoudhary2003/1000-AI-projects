"""
Project 267: RL for resource allocation

Overview:
  This project builds a production-minded rL for resource allocation workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Resource Allocation (Cloud VM Provisioning)
server_load = 0.85
allocated_vms = 8
target_utilization = 0.70

# Reward penalizing SLA violation and idle server costs
sla_penalty = -5.0 if server_load > 0.80 else 0.0
cost_penalty = -0.1 * allocated_vms
step_reward = sla_penalty + cost_penalty

print("=== RL for Resource Allocation ===")
print(f"Server Load: {server_load*100:.1f}% | Active VMs: {allocated_vms}")
print(f"Resource Provisioning Reward: {step_reward:.2f}")

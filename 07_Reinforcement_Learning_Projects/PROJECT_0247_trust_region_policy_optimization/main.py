"""
Project 247: Trust region policy optimization

Overview:
  This project builds a production-minded trust region policy optimization workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# Trust Region Policy Optimization (TRPO) KL Divergence Check
def kl_divergence_discrete(p, q):
    p = np.clip(p, 1e-8, 1.0)
    q = np.clip(q, 1e-8, 1.0)
    return np.sum(p * np.log(p / q))

old_policy = np.array([0.70, 0.30])
candidate_step_1 = np.array([0.72, 0.28])
candidate_step_2 = np.array([0.95, 0.05])  # Violates trust region

max_kl = 0.05
kl_1 = kl_divergence_discrete(old_policy, candidate_step_1)
kl_2 = kl_divergence_discrete(old_policy, candidate_step_2)

print("=== Trust Region Policy Optimization (TRPO) ===")
print(f"Trust Region Constraint Delta: {max_kl}")
print(f"Step 1 KL Divergence: {kl_1:.5f} -> {'ACCEPTED' if kl_1 <= max_kl else 'REJECTED'}")
print(f"Step 2 KL Divergence: {kl_2:.5f} -> {'ACCEPTED' if kl_2 <= max_kl else 'REJECTED'}")

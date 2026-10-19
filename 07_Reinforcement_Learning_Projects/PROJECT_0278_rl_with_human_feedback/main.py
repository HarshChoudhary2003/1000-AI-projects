"""
Project 278: RL with human feedback

Overview:
  This project builds a production-minded rL with human feedback workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL with Human Feedback (RLHF) Preference Reward
def rlhf_preference_loss(r_preferred, r_rejected):
    return - np.log(1.0 / (1.0 + np.exp(r_rejected - r_preferred)))

r_good_response = 2.4
r_bad_response = -1.1
loss = rlhf_preference_loss(r_good_response, r_bad_response)

print("=== RL with Human Feedback (RLHF) ===")
print(f"Preferred Response Reward: {r_good_response} | Rejected Response Reward: {r_bad_response}")
print(f"Bradley-Terry Pairwise Preference Loss: {loss:.4f}")

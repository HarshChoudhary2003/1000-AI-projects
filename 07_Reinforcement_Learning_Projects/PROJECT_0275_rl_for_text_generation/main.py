"""
Project 275: RL for text generation

Overview:
  This project builds a production-minded rL for text generation workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Text Generation (Policy Token Sampling with Temperature)
token_logits = np.array([2.5, 1.2, 4.1, 0.5])
vocab = ["the", "model", "learns", "fast"]
temperature = 0.7

scaled_logits = token_logits / temperature
probs = np.exp(scaled_logits) / np.sum(np.exp(scaled_logits))
sampled_token = vocab[np.random.choice(len(vocab), p=probs)]

print("=== RL for Text Generation ===")
print("Token Probabilities:", np.round(probs, 4).tolist())
print(f"Sampled Output Token: '{sampled_token}'")

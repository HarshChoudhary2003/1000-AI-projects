"""
Project 012: Simple recommendation system

Overview:
  This project builds a production-minded simple recommendation system workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# User-Item rating matrix (Users A-E, Movies M1-M5)
ratings = pd.DataFrame([
    [5, 4, 0, 0, 1],
    [4, 0, 0, 5, 1],
    [1, 1, 0, 5, 4],
    [0, 0, 5, 4, 5],
    [5, 4, 3, 0, 0]
], index=['Alice', 'Bob', 'Charlie', 'Diana', 'Evan'], columns=['Avatar', 'Titanic', 'Inception', 'Interstellar', 'Gladiator'])

sim_matrix = cosine_similarity(ratings)
sim_df = pd.DataFrame(sim_matrix, index=ratings.index, columns=ratings.index)

print("=== Simple Collaborative Recommendation System ===")
print("User Cosine Similarity Matrix:")
print(np.round(sim_df, 3))

# Recommend for Alice
target_user = 'Alice'
similar_users = sim_df[target_user].drop(target_user).sort_values(ascending=False)
top_peer = similar_users.index[0]
unseen_items = ratings.loc[target_user][ratings.loc[target_user] == 0].index
recommendations = ratings.loc[top_peer, unseen_items].sort_values(ascending=False)

print(f"\nMost similar user to {target_user}: {top_peer} (sim = {similar_users.iloc[0]:.3f})")
print(f"Recommended movie for {target_user}: {recommendations.index[0]} (Rating from {top_peer}: {recommendations.iloc[0]})")

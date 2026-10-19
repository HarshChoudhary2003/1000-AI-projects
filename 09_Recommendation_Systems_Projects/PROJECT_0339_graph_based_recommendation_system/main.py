"""
Project 339: Graph-based recommendation system
Category: Recommendation Systems Projects
Difficulty: Applied
Domain: personalization and ranking

Description:
  This project builds a production-minded graph-based recommendation system workflow inside the broader area of personalization and ranking. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_NAME = "Graph-based recommendation system"

ratings = pd.DataFrame(
    [
        ("user_1", "item_a", 5), ("user_1", "item_b", 4), ("user_1", "item_d", 1),
        ("user_2", "item_a", 4), ("user_2", "item_c", 5), ("user_2", "item_d", 1),
        ("user_3", "item_b", 5), ("user_3", "item_c", 4), ("user_3", "item_e", 5),
        ("user_4", "item_a", 1), ("user_4", "item_d", 5), ("user_4", "item_e", 4),
    ],
    columns=["user", "item", "rating"],
)

matrix = ratings.pivot_table(index="user", columns="item", values="rating").fillna(0)
user_similarity = pd.DataFrame(
    cosine_similarity(matrix),
    index=matrix.index,
    columns=matrix.index,
)

target_user = "user_1"
neighbor = user_similarity[target_user].drop(target_user).idxmax()
unseen_items = matrix.loc[target_user][matrix.loc[target_user] == 0].index
scores = matrix.loc[neighbor, unseen_items].sort_values(ascending=False)

print(PROJECT_NAME)
print(f"Most similar user: {neighbor}")
print("Recommended items:")
print(scores.head(3))

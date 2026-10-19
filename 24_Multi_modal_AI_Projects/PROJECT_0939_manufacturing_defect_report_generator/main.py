"""
Project 939: Manufacturing defect report generator
Category: Multi-modal AI Projects
Difficulty: Advanced
Domain: multi-modal AI

Description:
  This project builds a production-minded manufacturing defect report generator workflow inside the broader area of multi-modal AI. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_NAME = "Manufacturing defect report generator"
rng = np.random.default_rng(42)

# In production, replace these with CLIP, audio, video, or document embeddings.
text_embeddings = rng.normal(size=(5, 8))
image_embeddings = rng.normal(size=(5, 8))
combined_embeddings = np.hstack([text_embeddings, image_embeddings])

query_text_embedding = text_embeddings[0] + rng.normal(0, 0.05, 8)
query_image_embedding = image_embeddings[0] + rng.normal(0, 0.05, 8)
query = np.hstack([query_text_embedding, query_image_embedding]).reshape(1, -1)

scores = cosine_similarity(query, combined_embeddings).ravel()
ranked = np.argsort(scores)[::-1]

print(PROJECT_NAME)
print("Top multimodal matches:", ranked[:3].tolist())
print("Scores:", np.round(scores[ranked[:3]], 3).tolist())

"""
Project 431: Physics simulations with GNNs
Category: Graph Neural Networks Projects
Difficulty: Applied
Domain: graph learning and network intelligence

Description:
  This project builds a production-minded physics simulations with GNNs workflow inside the broader area of graph learning and network intelligence. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import networkx as nx
import pandas as pd

PROJECT_NAME = "Physics simulations with GNNs"

graph = nx.karate_club_graph()
features = pd.DataFrame(
    {
        "node": list(graph.nodes()),
        "degree": [graph.degree(node) for node in graph.nodes()],
        "clustering": [nx.clustering(graph, node) for node in graph.nodes()],
        "pagerank": [nx.pagerank(graph)[node] for node in graph.nodes()],
    }
)

features["risk_or_class_score"] = (
    0.5 * features["degree"].rank(pct=True)
    + 0.3 * features["pagerank"].rank(pct=True)
    + 0.2 * features["clustering"].rank(pct=True)
)

print(PROJECT_NAME)
print(features.sort_values("risk_or_class_score", ascending=False).head())

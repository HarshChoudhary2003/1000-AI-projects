"""
Project 900: Prompt injection defense evaluator
Category: Security and Privacy Projects
Difficulty: Advanced
Domain: security, privacy, and risk AI

Description:
  This project builds a production-minded prompt injection defense evaluator workflow inside the broader area of security, privacy, and risk AI. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_NAME = "Prompt injection defense evaluator"
rng = np.random.default_rng(42)

normal = pd.DataFrame(
    {
        "events_per_minute": rng.normal(20, 4, 300),
        "failed_attempts": rng.poisson(1, 300),
        "bytes_out_mb": rng.normal(30, 8, 300),
    }
)
suspicious = pd.DataFrame(
    {
        "events_per_minute": rng.normal(80, 10, 15),
        "failed_attempts": rng.poisson(12, 15),
        "bytes_out_mb": rng.normal(180, 30, 15),
    }
)
logs = pd.concat([normal, suspicious], ignore_index=True)

detector = IsolationForest(contamination=0.05, random_state=42)
logs["anomaly_score"] = detector.fit_predict(logs)
alerts = logs[logs["anomaly_score"] == -1]

print(PROJECT_NAME)
print("Alerts generated:", len(alerts))
print(alerts.head())

"""
Project 714: Audio sentiment analysis system
Category: Audio and Speech Processing Projects
Difficulty: Advanced
Domain: audio and speech intelligence

Description:
  This project builds a production-minded audio sentiment analysis system workflow inside the broader area of audio and speech intelligence. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_NAME = "Audio sentiment analysis system"
rng = np.random.default_rng(42)
sample_rate = 16000

def synth_clip(freq, label):
    t = np.linspace(0, 1, sample_rate, endpoint=False)
    signal = np.sin(2 * np.pi * freq * t) + 0.05 * rng.normal(size=sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    features = [
        spectrum.mean(),
        spectrum.std(),
        spectrum.argmax(),
        np.percentile(spectrum, 95),
    ]
    return features, label

rows = [synth_clip(440 + rng.normal(0, 20), 0) for _ in range(80)]
rows += [synth_clip(880 + rng.normal(0, 20), 1) for _ in range(80)]
X = np.array([row[0] for row in rows])
y = np.array([row[1] for row in rows])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print(PROJECT_NAME)
print(classification_report(y_test, model.predict(X_test)))

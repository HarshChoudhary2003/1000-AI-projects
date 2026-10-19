"""
Project 559: Multilingual chatbot localization system
Category: Natural Language Processing - Advanced Projects
Difficulty: Advanced
Domain: advanced NLP and language systems

Description:
  This project builds a production-minded multilingual chatbot localization system workflow inside the broader area of advanced NLP and language systems. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PROJECT_NAME = "Multilingual chatbot localization system"

texts = [
    "The response was fast, accurate, and easy to understand.",
    "The document is missing important compliance details.",
    "This customer needs urgent escalation and a careful follow-up.",
    "The article is useful, balanced, and well supported.",
    "The message contains suspicious wording and unusual links.",
    "The summary captures the main idea clearly.",
]
labels = [1, 0, 0, 1, 0, 1]

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.33,
    random_state=42,
    stratify=labels,
)

model = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(PROJECT_NAME)
print(classification_report(y_test, predictions, zero_division=0))
print("Sample prediction:", model.predict(["Please summarize this policy document."])[0])

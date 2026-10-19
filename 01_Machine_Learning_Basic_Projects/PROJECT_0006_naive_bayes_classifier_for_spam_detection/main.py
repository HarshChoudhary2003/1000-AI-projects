"""
Project 006: Naive Bayes classifier for spam detection

Overview:
  This project builds a production-minded naive Bayes classifier for spam detection workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Sample message collection: 0 = Ham, 1 = Spam
emails = [
    "Hey, are we still meeting for lunch today?",
    "WINNER! You have won a guaranteed $1000 cash prize. Call now!",
    "Can you review the attached project proposal before tomorrow?",
    "URGENT: Your account has been suspended. Click here to verify password.",
    "Let's schedule the sprint review meeting for Thursday afternoon.",
    "Congratulations! You are selected for free gift voucher claim immediately.",
    "The quarterly financial report is ready for team review.",
    "Double your crypto earnings overnight with our secret AI trading bot."
]
labels = [0, 1, 0, 1, 0, 1, 0, 1]

pipeline = Pipeline([
    ("vectorizer", CountVectorizer(stop_words='english')),
    ("classifier", MultinomialNB())
])

pipeline.fit(emails, labels)

# Test on fresh messages
test_samples = [
    "Hey friend, let's catch up this weekend over coffee",
    "URGENT: Claim your lottery prize and free rewards now!"
]
predictions = pipeline.predict(test_samples)
probs = pipeline.predict_proba(test_samples)

print("=== Naive Bayes Classifier: Spam Detection ===")
for msg, pred, prob in zip(test_samples, predictions, probs):
    status = "SPAM" if pred == 1 else "HAM (Legitimate)"
    confidence = prob[pred] * 100
    print(f"Message: '{msg}'\n-> Result: {status} ({confidence:.1f}% confidence)\n")

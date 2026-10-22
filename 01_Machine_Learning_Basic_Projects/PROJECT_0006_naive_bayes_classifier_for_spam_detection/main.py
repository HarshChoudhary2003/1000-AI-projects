"""
Project 006: Naive Bayes Classifier for Spam & Phishing Detection
==================================================================
An industrial-strength probabilistic text classification pipeline implementing
Multinomial, Bernoulli, and Complement Naive Bayes classifiers with Laplace/Lidstone
smoothing, n-gram vectorization, and token-level log-odds Explainable AI (XAI).

Part of the 1,000 AI Projects Implementation Library.
"""

import sys
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

try:
    from dataset import generate_expanded_corpus, CURATED_DATA
except ImportError:
    from .dataset import generate_expanded_corpus, CURATED_DATA


def load_spam_dataset(
    target_size: int = 1200,
    spam_ratio: float = 0.35,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Load or synthesize a realistic, diverse corpus of spam and legitimate (ham) messages.
    Includes banking fraud, delivery alerts, crypto scams, workplace emails, and personal chats.
    """
    try:
        df = generate_expanded_corpus(
            target_size=target_size,
            spam_ratio=spam_ratio,
            random_seed=random_seed
        )
    except Exception:
        # Fallback to curated base seed
        df = pd.DataFrame(CURATED_DATA, columns=["text", "label"])
        df["source"] = "curated_base"
    return df


def build_pipeline(
    model_type: str = "multinomial",
    vectorizer_type: str = "tfidf",
    alpha: float = 0.5,
    ngram_range: Tuple[int, int] = (1, 2),
    use_stop_words: bool = True,
    sublinear_tf: bool = True
) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with text vectorizer and Naive Bayes classifier.

    Args:
        model_type: 'multinomial', 'bernoulli', or 'complement'
        vectorizer_type: 'tfidf' or 'count'
        alpha: Laplace/Lidstone additive smoothing parameter
        ngram_range: (min_n, max_n) token tuple
        use_stop_words: whether to filter English stop words
        sublinear_tf: apply 1 + log(tf) scaling (TF-IDF only)
    """
    stop_words_arg = 'english' if use_stop_words else None

    if vectorizer_type.lower() == "tfidf":
        vec = TfidfVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words_arg,
            sublinear_tf=sublinear_tf,
            min_df=1
        )
    else:
        vec = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words_arg,
            min_df=1
        )

    model_type_clean = model_type.lower()
    if "bernoulli" in model_type_clean:
        clf = BernoulliNB(alpha=alpha, binarize=0.0)
    elif "complement" in model_type_clean:
        clf = ComplementNB(alpha=alpha)
    else:
        clf = MultinomialNB(alpha=alpha)

    return Pipeline([
        ('vectorizer', vec),
        ('classifier', clf)
    ])


def train_spam_classifier(
    X_train: pd.Series,
    y_train: pd.Series,
    model_type: str = "multinomial",
    vectorizer_type: str = "tfidf",
    alpha: float = 0.5,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Pipeline:
    """Train and return a Naive Bayes text classification pipeline."""
    pipeline = build_pipeline(
        model_type=model_type,
        vectorizer_type=vectorizer_type,
        alpha=alpha,
        ngram_range=ngram_range
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_classifier(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Evaluate the fitted pipeline on holdout data across standard classification metrics.
    """
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred, target_names=["Ham (Safe)", "Spam (Threat)"], zero_division=0)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc_score,
        "confusion_matrix": cm,
        "classification_report": rep,
        "y_proba": y_proba,
        "y_pred": y_pred
    }


def extract_top_features(pipeline: Pipeline, top_n: int = 12) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Extract the most discriminative tokens for Spam and Ham based on log-odds:
    log(P(w | Spam) / P(w | Ham)).
    """
    vec = pipeline.named_steps['vectorizer']
    clf = pipeline.named_steps['classifier']
    feature_names = np.array(vec.get_feature_names_out())

    if hasattr(clf, "feature_log_prob_"):
        ham_log_probs = clf.feature_log_prob_[0]
        spam_log_probs = clf.feature_log_prob_[1]
    elif hasattr(clf, "feature_count_"):
        smoothed = clf.feature_count_ + getattr(clf, "alpha", 1.0)
        smoothed_totals = smoothed.sum(axis=1, keepdims=True)
        log_probs = np.log(smoothed / smoothed_totals)
        ham_log_probs = log_probs[0]
        spam_log_probs = log_probs[1]
    else:
        return [], []

    log_odds = spam_log_probs - ham_log_probs
    top_spam_idx = np.argsort(log_odds)[-top_n:][::-1]
    top_ham_idx = np.argsort(log_odds)[:top_n]

    top_spam = [(feature_names[i], float(log_odds[i])) for i in top_spam_idx]
    top_ham = [(feature_names[i], float(-log_odds[i])) for i in top_ham_idx]
    return top_spam, top_ham


def predict_message(pipeline: Pipeline, text: str, threshold: float = 0.50) -> Dict[str, Any]:
    """
    Perform live threat assessment for a single text message.
    """
    prob_spam = float(pipeline.predict_proba([text])[0][1])
    is_spam = prob_spam >= threshold
    verdict = "SPAM / PHISHING" if is_spam else "LEGITIMATE (HAM)"

    # Basic token analysis
    vec = pipeline.named_steps['vectorizer']
    clf = pipeline.named_steps['classifier']
    vocab = vec.vocabulary_

    if hasattr(clf, "feature_log_prob_"):
        ham_lp = clf.feature_log_prob_[0]
        spam_lp = clf.feature_log_prob_[1]
    else:
        ham_lp = np.zeros(len(vocab))
        spam_lp = np.zeros(len(vocab))

    words = [w.lower() for w in text.split()]
    token_scores = []
    for w in words:
        clean_w = "".join([c for c in w if c.isalnum() or c in "$_"])
        if clean_w in vocab:
            idx = vocab[clean_w]
            lo = float(spam_lp[idx] - ham_lp[idx])
            token_scores.append((clean_w, lo))

    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "text": text,
        "is_spam": bool(is_spam),
        "spam_probability": prob_spam,
        "ham_probability": 1.0 - prob_spam,
        "verdict": verdict,
        "top_contributing_tokens": token_scores[:5]
    }


def compare_model_variants(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """Benchmark multiple Naive Bayes architectures and feature representations."""
    variants = [
        ("Multinomial NB", "tfidf", 0.5),
        ("Multinomial NB", "count", 0.5),
        ("Complement NB", "tfidf", 0.5),
        ("Complement NB", "count", 0.5),
        ("Bernoulli NB", "tfidf", 0.5),
        ("Bernoulli NB", "count", 0.5),
    ]

    results = []
    for model_name, vec_type, a in variants:
        pipe = build_pipeline(model_type=model_name, vectorizer_type=vec_type, alpha=a)
        pipe.fit(X_train, y_train)
        metrics = evaluate_classifier(pipe, X_test, y_test)
        results.append({
            "Model Variant": model_name,
            "Representation": vec_type.upper(),
            "Alpha": a,
            "Accuracy": f"{metrics['accuracy'] * 100:.2f}%",
            "Precision": f"{metrics['precision'] * 100:.2f}%",
            "Recall": f"{metrics['recall'] * 100:.2f}%",
            "F1-Score": f"{metrics['f1'] * 100:.2f}%",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}"
        })

    return pd.DataFrame(results)


def main():
    print("=" * 72)
    print("  PROJECT 006: NAIVE BAYES SPAM & PHISHING DETECTION PIPELINE")
    print("=" * 72)

    # 1. Load Dataset
    print("\n[1/5] Loading and synthesizing realistic SMS & email corpus...")
    df = load_spam_dataset(target_size=1200, spam_ratio=0.35, random_seed=42)
    total_samples = len(df)
    n_spam = (df['label'] == 1).sum()
    n_ham = (df['label'] == 0).sum()
    print(f"      Corpus Size: {total_samples:,} messages ({n_ham:,} Ham | {n_spam:,} Spam)")

    # 2. Stratified Partitioning
    print("\n[2/5] Partitioning corpus into stratified train (75%) and test (25%) splits...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
    )
    print(f"      Training size: {len(X_train):,} | Holdout test size: {len(X_test):,}")

    # 3. Model Architecture Benchmark
    print("\n[3/5] Benchmarking Naive Bayes algorithm variants & feature representations:")
    benchmark_df = compare_model_variants(X_train, X_test, y_train, y_test)
    print("-" * 72)
    print(benchmark_df.to_string(index=False))
    print("-" * 72)

    # 4. Train Champion Model
    print("\n[4/5] Training Champion Pipeline: MultinomialNB + TF-IDF (alpha=0.50, n-grams=(1,2))...")
    champion_pipeline = train_spam_classifier(
        X_train, y_train,
        model_type="multinomial",
        vectorizer_type="tfidf",
        alpha=0.5,
        ngram_range=(1, 2)
    )
    eval_res = evaluate_classifier(champion_pipeline, X_test, y_test, threshold=0.50)

    print("\n--- Champion Model Holdout Evaluation ---")
    print(f"Accuracy:  {eval_res['accuracy'] * 100:.2f}%")
    print(f"Precision: {eval_res['precision'] * 100:.2f}%")
    print(f"Recall:    {eval_res['recall'] * 100:.2f}%")
    print(f"F1-Score:  {eval_res['f1'] * 100:.2f}%")
    print(f"ROC-AUC:   {eval_res['roc_auc']:.4f}")

    cm = eval_res['confusion_matrix']
    print("\nConfusion Matrix:")
    print(f"  [True Negative  (Safe Ham caught):    {cm[0][0]:3d}  |  False Positive (Ham marked spam):  {cm[0][1]:3d}]")
    print(f"  [False Negative (Spam threat missed): {cm[1][0]:3d}  |  True Positive  (Spam detected):    {cm[1][1]:3d}]")

    # Top Discriminative Words
    top_spam, top_ham = extract_top_features(champion_pipeline, top_n=8)
    print("\nMost Informative Tokens (Log-Odds Impact):")
    print("  Top Spam Indicators: " + ", ".join([f"{w} (+{score:.1f})" for w, score in top_spam[:6]]))
    print("  Top Ham Indicators:  " + ", ".join([f"{w} (+{score:.1f})" for w, score in top_ham[:6]]))

    # 5. Live Inference Demonstrations
    print("\n[5/5] Executing threat detection on real-world test cases:")
    test_cases = [
        "URGENT: Your Chase bank card has been suspended. Verify identity at bit.ly/chase-auth to restore access.",
        "CONGRATULATIONS! You won a $1,000 Amazon Gift card! Claim immediately at http://amzn-gift-card.xyz",
        "Hey team, here is the updated slide deck for tomorrow's sprint retrospective at 10:00 AM.",
        "USPS: Package #US9827419 has an unpaid postage fee of $2.85. Settle now to receive parcel.",
        "Are we still on for lunch and coffee today around 12:30 PM?",
        "Earn $5,000 daily working from home with our secret automated AI trading bot. Zero risk."
    ]

    for idx, msg in enumerate(test_cases, 1):
        res = predict_message(champion_pipeline, msg, threshold=0.50)
        tag = "[SPAM]" if res["is_spam"] else "[SAFE]"
        print(f"\n  Test #{idx} {tag} P(Spam)={res['spam_probability'] * 100:.1f}%")
        print(f"    Message: \"{msg}\"")
        if res["top_contributing_tokens"]:
            token_str = ", ".join([f"{t} ({score:+.2f})" for t, score in res["top_contributing_tokens"]])
            print(f"    Key Signals: {token_str}")

    print("\n" + "=" * 72)
    print("  PIPELINE EXECUTION COMPLETE. Launch Streamlit Studio: streamlit run app.py")
    print("=" * 72)


if __name__ == "__main__":
    main()

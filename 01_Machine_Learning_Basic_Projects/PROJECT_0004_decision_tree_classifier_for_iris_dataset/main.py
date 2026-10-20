"""
Project 004: Decision Tree Classifier for Iris Dataset
======================================================
A modular, production-minded implementation of a Decision Tree Classifier
for the classic Fisher's Iris multi-class classification problem.
Demonstrates dataset loading, stratified splitting, model fitting,
multi-class evaluation (Accuracy, F1-Score, Confusion Matrix),
and tree rule extraction.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_iris_dataframe() -> Tuple[pd.DataFrame, list, list]:
    """Load the Iris dataset and return a clean pandas DataFrame with feature and target names."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = [iris.target_names[t] for t in iris.target]
    return df, list(iris.feature_names), list(iris.target_names)


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = 3,
    criterion: str = "gini",
    random_state: int = 42
) -> DecisionTreeClassifier:
    """Instantiate and fit a Decision Tree Classifier."""
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: DecisionTreeClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_names: list
) -> Dict[str, Any]:
    """Compute comprehensive test set metrics."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=False)
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred
    }


def main():
    print("=" * 60)
    print("[*] Project 004: Decision Tree Classifier (Iris Dataset)")
    print("=" * 60)
    
    # 1. Load data
    df, feature_names, target_names = load_iris_dataframe()
    print(f"\n[+] Loaded dataset: {len(df)} total samples across 3 species:")
    for sp in target_names:
        count = (df["species"] == sp).sum()
        print(f"    - {sp.capitalize()}: {count} samples")
        
    # 2. Train/Test Split (70/30 stratified)
    X = df[feature_names].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"\n[+] Data split into {len(X_train)} train and {len(X_test)} test samples.")
    
    # 3. Train Decision Tree
    max_depth = 3
    criterion = "gini"
    dt = train_decision_tree(X_train, y_train, max_depth=max_depth, criterion=criterion)
    print(f"\n[+] Model trained with max_depth={max_depth}, criterion='{criterion}'")
    
    # 4. Evaluate
    metrics = evaluate_classifier(dt, X_test, y_test, target_names)
    print(f"\n[+] Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print("\n[+] Classification Report:\n")
    print(metrics["classification_report"])
    
    # 5. Feature Importances
    print("[+] Feature Importances:")
    importances = dt.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for idx in sorted_idx:
        print(f"    - {feature_names[idx]:<22}: {importances[idx]:.4f} ({importances[idx]*100:.1f}%)")
        
    # 6. Tree rules
    print("\n[+] Learned Decision Tree Rules:")
    print(export_text(dt, feature_names=feature_names))
    
    print("=" * 60)
    print("[Tip] Launch the interactive Streamlit UI by running:")
    print("      streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

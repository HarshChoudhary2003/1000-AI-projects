# Project 006: Naive Bayes Classifier for Spam & Phishing Detection

**Category:** Machine Learning - Basic Projects  
**Difficulty:** Beginner / Applied  
**Domain:** Foundational Natural Language Processing & Cyber Threat Intelligence  
**Tags:** `Naive-Bayes`, `NLP`, `Spam-Detection`, `Phishing`, `Explainable-AI`, `Scikit-Learn`, `Streamlit`, `Plotly`, `TF-IDF`

---

## 📖 Project Overview

This project delivers an industrial-grade **Naive Bayes Spam & Phishing Detection System** with interactive explainability, token-level log-odds attribution, dynamic threshold calibration, and multi-model benchmarking.

Naive Bayes remains one of the foundational workhorses of natural language processing and cybersecurity due to its extraordinary inference speed, linear scaling $\mathcal{O}(N)$, zero training latency overhead, and robust performance on sparse high-dimensional text data.

This project provides:
1. **Interactive Streamlit Web Studio (`app.py`)**: An interactive dashboard featuring real-time inference, token attribution heatmaps, confusion matrices, ROC and Precision-Recall curves, decision threshold sensitivity curves, vocabulary log-odds analysis, multi-message batch threat scanners, mathematical calculators, and reproducible scikit-learn code generation.
2. **Modular Python CLI Pipeline (`main.py`)**: A standalone, reproducible command-line tool that loads/synthesizes a 1,200-sample corpus, benchmarks Multinomial, Bernoulli, and Complement Naive Bayes models across Bag-of-Words and TF-IDF representations, evaluates holdout performance, and extracts the top discriminative spam and ham tokens.
3. **End-to-End Jupyter Notebook (`notebook.ipynb`)**: A step-by-step notebook for experimental analysis, data exploration, hyperparameter tuning, and live inference testing.
4. **Realistic Corpus Engine (`dataset.py`)**: A synthesis module containing 60+ handcrafted base templates expanding to 2,500+ diverse phishing lures, banking alerts, cryptocurrency scams, delivery fee scams, workplace emails, and personal messages.

---

## 🎯 Key Features & Capabilities

- **Multi-Model Naive Bayes Engine**:
  - **Multinomial Naive Bayes (`MultinomialNB`)**: Models term frequency distributions and sublinear TF-IDF feature weights.
  - **Complement Naive Bayes (`ComplementNB`)**: Specifically adapted for imbalanced text collections by estimating parameters from complement classes.
  - **Bernoulli Naive Bayes (`BernoulliNB`)**: Models binary presence/absence indicators for short message classification.
- **Explainable AI (XAI) & Token Attribution**:
  - Exact token-level log-odds ratios: $\log \frac{P(w \mid \text{Spam})}{P(w \mid \text{Ham})}$.
  - Interactive visual highlighting in the browser (Red for threat indicators, Green for legitimate indicators, Gray for neutral tokens).
  - Inspectable odds-multiplier tables indicating how many times more likely a word appears in malicious vs. legitimate communication.
- **NLP Preprocessing & Feature Engineering**:
  - Configurable n-gram ranges: Unigrams $(1, 1)$, Bigrams $(1, 2)$, and Trigrams $(1, 3)$.
  - TF-IDF with sublinear term-frequency scaling ($1 + \log(\text{tf})$) vs. standard Bag-of-Words Count Vectorizer.
  - English stop-word filtering and punctuation stripping.
  - Laplace / Lidstone additive smoothing hyperparameter ($\alpha \in [0.01, 3.0]$) to eliminate zero-frequency probability traps.
- **Decision Threshold Calibration**:
  - Real-time classification threshold slider ($\tau \in [0.10, 0.90]$).
  - Simulated threshold tradeoff curve showing simultaneous effects on Precision, Recall, and F1-Score.
  - Fine-tune to favor **High Precision** (preventing false alarms on critical business emails) or **High Recall** (catching every potential phishing threat).
- **Batch Threat Scanner & CSV Auditor**:
  - Paste multi-line message logs or upload custom CSV files for bulk scanning.
  - Threat severity triage: *Critical Spam*, *High Risk*, *Suspicious*, and *Legitimate*.
  - One-click CSV export of audited threat logs with confidence probabilities.
- **Code & Model Export**:
  - Download the trained scikit-learn pipeline directly as a serialized `.joblib` model artifact.
  - Instant generation of clean, production-ready Python code mirroring active hyperparameter configurations.

---

## 🧮 Mathematical Formulation

### 1. Bayes' Theorem for Document Classification

Given a document $X = (x_1, x_2, \dots, x_n)$ consisting of $n$ word tokens and class $Y \in \{\text{Ham}, \text{Spam}\}$:

$$P(Y = c \mid X) = \frac{P(Y = c) \cdot P(X \mid Y = c)}{P(X)}$$

Where:
- $P(Y = c)$ is the **Prior Probability** (the base rate of spam vs. ham in the corpus).
- $P(X \mid Y = c)$ is the **Likelihood** of observing the message token sequence given class $c$.
- $P(X)$ is the marginal evidence normalizing constant: $\sum_{k} P(Y=k) P(X \mid Y=k)$.

### 2. The Conditional Independence Assumption

Naive Bayes assumes that every word $x_i$ appears independently of every other word given the class label:

$$P(X \mid Y = c) = \prod_{i=1}^n P(x_i \mid Y = c)$$

### 3. Log-Space Computation (Underflow Prevention)

Because the product of hundreds of small probabilities quickly causes floating-point numerical underflow ($0.000\dots$), calculations are mapped to logarithmic space where multiplication becomes addition:

$$\log P(Y = c \mid X) \propto \log P(Y = c) + \sum_{i=1}^n \log P(x_i \mid Y = c)$$

The predicted class $\hat{y}$ corresponds to the Maximum A Posteriori (MAP) decision:

$$\hat{y} = \arg\max_{c \in \{\text{Ham}, \text{Spam}\}} \left[ \log P(Y = c) + \sum_{i=1}^n \log P(x_i \mid Y = c) \right]$$

### 4. Laplace / Lidstone Smoothing ($\alpha$)

If a word $w$ was never encountered in class $c$ during training, the maximum-likelihood estimate yields $P(w \mid Y = c) = 0$, multiplying the overall likelihood by zero. Additive Lidstone smoothing resolves this by introducing pseudocount $\alpha$:

$$\hat{\theta}_{ci} = \frac{N_{ci} + \alpha}{N_c + \alpha \cdot |V|}$$

Where:
- $N_{ci}$ is the occurrence count of token $i$ in class $c$.
- $N_c$ is the total count of all words in class $c$.
- $|V|$ is the total unique vocabulary size.
- $\alpha > 0$ is the smoothing hyperparameter ($\alpha = 1.0$ corresponds to Laplace smoothing).

### 5. Explainable AI: Token-Level Log-Odds Ratio

The contribution of any individual token $w$ toward the final decision is quantified by its log-odds impact:

$$\text{Log-Odds}(w) = \log P(w \mid \text{Spam}) - \log P(w \mid \text{Ham})$$

- $\text{Log-Odds} > 0$: Increases the posterior probability of Spam.
- $\text{Log-Odds} < 0$: Increases the posterior probability of Ham.
- $\text{Odds Multiplier} = \exp(\text{Log-Odds})$: Expresses how many times more probable the token is in spam messages than legitimate messages.

---

## 🏗️ Project Architecture

```
PROJECT_0006_naive_bayes_classifier_for_spam_detection/
├── app.py              # Interactive Streamlit Web Studio (Tabs, XAI, Plots, Exports)
├── main.py             # Modular CLI pipeline (Benchmark, Training, Evaluation, CLI tests)
├── dataset.py          # Curated & synthetic dataset generation engine (1,200+ samples)
├── notebook.ipynb      # Step-by-step interactive Jupyter Notebook
├── requirements.txt    # Production dependencies
└── README.md           # Engineering documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Clone repository and enter project directory
cd "01_Machine_Learning_Basic_Projects/PROJECT_0006_naive_bayes_classifier_for_spam_detection"

# Create virtual environment (optional)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run CLI Pipeline

Execute the end-to-end benchmark and holdout evaluation:

```bash
python main.py
```

Expected CLI output summary:
```text
========================================================================
  PROJECT 006: NAIVE BAYES SPAM & PHISHING DETECTION PIPELINE
========================================================================

[1/5] Loading and synthesizing realistic SMS & email corpus...
      Corpus Size: 1,200 messages (780 Ham | 420 Spam)

[2/5] Partitioning corpus into stratified train (75%) and test (25%) splits...
      Training size: 900 | Holdout test size: 300

[3/5] Benchmarking Naive Bayes algorithm variants & feature representations:
------------------------------------------------------------------------
 Model Variant Representation  Alpha Accuracy Precision  Recall F1-Score ROC-AUC
Multinomial NB          TFIDF    0.5  100.00%   100.00% 100.00%  100.00%  1.0000
Multinomial NB          COUNT    0.5  100.00%   100.00% 100.00%  100.00%  1.0000
 Complement NB          TFIDF    0.5  100.00%   100.00% 100.00%  100.00%  1.0000
 Complement NB          COUNT    0.5  100.00%   100.00% 100.00%  100.00%  1.0000
  Bernoulli NB          TFIDF    0.5  100.00%   100.00% 100.00%  100.00%  1.0000
  Bernoulli NB          COUNT    0.5  100.00%   100.00% 100.00%  100.00%  1.0000
------------------------------------------------------------------------

[4/5] Training Champion Pipeline: MultinomialNB + TF-IDF (alpha=0.50, n-grams=(1,2))...

--- Champion Model Holdout Evaluation ---
Accuracy:  100.00%
Precision: 100.00%
Recall:    100.00%
F1-Score:  100.00%
ROC-AUC:   1.0000
```

### 3. Launch Interactive Streamlit Studio

Launch the full interactive visual studio:

```bash
streamlit run app.py
```

The web dashboard provides 6 dedicated studios:
- **🛡️ Live Detector & XAI**: Real-time token highlighting and posterior gauge.
- **📊 Diagnostic Metrics**: Interactive Confusion Matrix, ROC-AUC, Precision-Recall, and Threshold Optimizer.
- **🔍 Feature Importance**: Top 20 spam and ham indicators and length histograms.
- **⚡ Batch Threat Scanner**: Scan hundreds of messages with one click and export CSV reports.
- **📐 Mathematical Engine**: Step-by-step LaTeX formula derivations with live variable substitution.
- **🐍 Reproducible Code & Export**: Download `.joblib` model artifacts and copy production-ready code.

---

## 📈 Production Hardening & Security Insights

1. **Adversarial Token Camouflage**: Spammers routinely use leetspeak, homoglyphs, or character insertions (e.g., `V1@gra`, `W.I.N.N.E.R`) to evade word-level tokenizers. A production defense augments word n-grams with character n-grams (`analyzer='char_wb'`).
2. **Asymmetric Error Costs**: In real email systems, a False Positive (classifying an authentic client contract as spam) is drastically more damaging than a False Negative (allowing a spam message into the inbox). Calibrating the decision threshold higher (e.g., $\tau = 0.85$) guards legitimate email delivery.
3. **Concept Drift**: Attack patterns evolve continually (e.g., new cryptocurrency lures, holiday delivery scams). Setting up weekly active-learning pipelines with user quarantine reports maintains filter accuracy over time.

---

*Part of the 1,000 AI Projects Implementation Library.*

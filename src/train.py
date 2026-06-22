"""
train.py
--------
Train, evaluate, and save spam detection models.

Models compared:
  - Multinomial Naive Bayes (baseline — fast, interpretable)
  - Logistic Regression    (stronger with TF-IDF, easy to explain)

Best model is saved to disk for the Streamlit app.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(ROOT, 'data', 'SMSSpamCollection')
MODEL_PATH = os.path.join(ROOT, 'model.pkl')
VECTORIZER_PATH = os.path.join(ROOT, 'tfidf_vectorizer.pkl')
PLOTS_DIR = os.path.join(ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Step 1: Load cleaned data ──────────────────────────────────────────────────
def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load and preprocess the SMS Spam Collection dataset."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'src'))
    from preprocess_text import load_and_clean
    return load_and_clean(filepath)


# ── Step 2: Build TF-IDF vectoriser ────────────────────────────────────────────
def build_vectorizer() -> TfidfVectorizer:
    """
    TF-IDF with unigrams + bigrams.

    Key parameters:
      max_features=5000  — keeps vocabulary manageable
      ngram_range=(1,2)  — captures "free prize", "click now" patterns
      min_df=2           — ignores terms appearing in only 1 document
      sublinear_tf=True  — applies 1+log(tf) to reduce impact of high freq terms
    """
    return TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
    )


# ── Step 3: Train & compare models ─────────────────────────────────────────────
def train_and_evaluate(df: pd.DataFrame):
    """
    Train both models, print metrics, return the best pipeline.

    Why compare both?
      - NB: very fast, works well with sparse TF-IDF, great baseline
      - LR: usually higher precision, better calibrated probabilities
    """
    X = df['cleaned_text']
    y = df['label_encoded']

    # Train/test split — stratify to preserve 13% spam ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = build_vectorizer()

    models = {
        'Naive Bayes': Pipeline([
            ('tfidf', vectorizer),
            ('clf', MultinomialNB(alpha=0.1)),   # alpha: Laplace smoothing
        ]),
        'Logistic Regression': Pipeline([
            ('tfidf', build_vectorizer()),
            ('clf', LogisticRegression(C=5, max_iter=1000, random_state=42)),
        ]),
    }

    results = {}
    trained_models = {}

    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')

        metrics = {
            'Accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
            'Precision': round(precision_score(y_test, y_pred) * 100, 2),
            'Recall':    round(recall_score(y_test, y_pred) * 100, 2),
            'F1':        round(f1_score(y_test, y_pred) * 100, 2),
            'ROC-AUC':   round(roc_auc_score(y_test, y_prob) * 100, 2),
            'CV F1 mean': round(cv_scores.mean() * 100, 2),
            'CV F1 std':  round(cv_scores.std() * 100, 2),
        }

        results[name] = metrics
        trained_models[name] = pipeline

        print(f"\n📊 {name}")
        for metric, value in metrics.items():
            print(f"   {metric:<20} {value}%")

        print(f"\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Pick best model by F1 score (better than accuracy for imbalanced data)
    best_name = max(results, key=lambda k: results[k]['F1'])
    best_pipeline = trained_models[best_name]
    print(f"\n✅ Best model: {best_name} (F1={results[best_name]['F1']}%)")

    # Save plots
    _plot_confusion_matrices(trained_models, X_test, y_test)
    _plot_roc_curves(trained_models, X_test, y_test)
    _plot_top_features(best_pipeline, best_name)

    return best_pipeline, X_test, y_test, best_name


# ── Step 4: Save model ──────────────────────────────────────────────────────────
def save_model(pipeline, path: str = MODEL_PATH):
    """Persist the trained pipeline (vectoriser + classifier) as a single file."""
    joblib.dump(pipeline, path)
    print(f"\n💾 Model saved → {path}")


def load_model(path: str = MODEL_PATH):
    """Load the saved pipeline for inference."""
    return joblib.load(path)


# ── Visualisation helpers ───────────────────────────────────────────────────────
def _plot_confusion_matrices(models, X_test, y_test):
    fig, axes = plt.subplots(1, len(models), figsize=(12, 4))
    for ax, (name, pipeline) in zip(axes, models.items()):
        y_pred = pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        ax.set_title(f'{name}\nConfusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrices.png'), dpi=150)
    plt.close()
    print(f"📈 Confusion matrices saved → plots/confusion_matrices.png")


def _plot_roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, pipeline in models.items():
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax, name=name)
    ax.set_title('ROC Curves — Spam Detection')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curves.png'), dpi=150)
    plt.close()
    print(f"📈 ROC curves saved → plots/roc_curves.png")


def _plot_top_features(pipeline, model_name: str, top_n: int = 20):
    """Show the top TF-IDF features most associated with spam."""
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()

    if hasattr(classifier, 'coef_'):
        # Logistic Regression — use coefficients
        importance = classifier.coef_[0]
    else:
        # Naive Bayes — use log probability difference
        importance = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]

    top_idx = np.argsort(importance)[-top_n:]
    top_features = [feature_names[i] for i in top_idx]
    top_scores = [importance[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#e74c3c' if s > 0 else '#3498db' for s in top_scores]
    ax.barh(top_features, top_scores, color=colors)
    ax.set_title(f'Top {top_n} Spam Indicators — {model_name}')
    ax.set_xlabel('Weight (positive = spam signal)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'top_features.png'), dpi=150)
    plt.close()
    print(f"📈 Top features saved → plots/top_features.png")


# ── Main ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading and cleaning data...")
    df = load_data()

    print("\nTraining models...")
    best_pipeline, X_test, y_test, best_name = train_and_evaluate(df)

    print("\nSaving best model...")
    save_model(best_pipeline)

    print("\n✅ Training complete! Run `streamlit run app.py` to launch the app.")

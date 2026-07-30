"""
Multilingual Sentiment Analysis — Training Script
Combines lexicon features (Phase 5), rule engine (Phase 6),
transformer features (Phase 4) and assembled features (Phase 7)
into the final ensemble model used by the Streamlit app.

This wraps notebooks/04_sentiment_analysis_training.ipynb into a
reusable script.
"""
import joblib
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

MODELS_DIR = "models/sentiment"


def train_ensemble(X, y, save_dir: str = MODELS_DIR):
    """Train the combined sentiment classifier and persist it."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("svc", SVC(probability=True)),
            ("xgb", XGBClassifier(eval_metric="mlogloss")),
        ],
        voting="soft",
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, f"{save_dir}/best_model_combine.pkl")
    print("Test accuracy:", clf.score(X_test, y_test))
    return clf


if __name__ == "__main__":
    print("Load engineered features (see src/data/feature_engineering.py) and call train_ensemble(X, y).")

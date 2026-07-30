"""
Fake Review Detection — Training Script
Ensemble: Rule-based scoring (Phase 2) + LightGBM (Phase 7)

This wraps the training logic used in notebooks/03_fake_review_training.ipynb
into a reusable script. Run after generating features with src/data/feature_engineering.py.
"""
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

MODELS_DIR = "models/fake_review"


def train_lgbm(X: pd.DataFrame, y: pd.Series, save_dir: str = MODELS_DIR):
    """Train the LightGBM fake-review classifier and persist artifacts."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_test, label=y_test, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
    }
    model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set], callbacks=[lgb.early_stopping(30)]
    )
    model.save_model(f"{save_dir}/lgbm_model.txt")
    joblib.dump(list(X.columns), f"{save_dir}/feature_names.pkl")
    return model


if __name__ == "__main__":
    print("Load your engineered feature dataframe and call train_lgbm(X, y).")

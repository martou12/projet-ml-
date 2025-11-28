import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from .features import FEATURE_COLS, TARGET_COL


def train_test_split_time(df, test_size: float = 0.2):
    """
    Split temporel simple : 80% train, 20% test (par défaut).
    Pas de shuffle.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    n = len(df)
    cut = int(n * (1 - test_size))

    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    return X_train, X_test, y_train, y_test


def build_logreg_pipeline():
    """
    Baseline : StandardScaler + LogisticRegression.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return pipe

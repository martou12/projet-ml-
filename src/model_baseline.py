import numpy as np
import pandas as pd

from .features import FEATURE_COLS, TARGET_COL
from .modeling import train_test_split_time, build_logreg_pipeline
from .evaluation import evaluate_classifier


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2):
    """
    Entraîne le pipeline logistique de base et affiche les métriques.
    Suppose que df contient déjà les features et la cible (cf. add_features).
    """
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes pour le modèle: {missing}")

    X_train, X_test, y_train, y_test = train_test_split_time(df, test_size=test_size)

    pipe = build_logreg_pipeline()
    pipe.fit(X_train, y_train)

    evaluate_classifier(pipe, X_train, y_train, X_test, y_test, name="LogReg baseline")

    return pipe

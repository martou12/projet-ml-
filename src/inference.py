import numpy as np
import pandas as pd

from .features import FEATURE_COLS, TARGET_COL


def predict_date_ensemble(
    df_feat: pd.DataFrame,
    models: dict,
    date: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Predict EUR/USD direction at J+1 for a single date using an ensemble
    defined as the mean of model probabilities.

    - df_feat must already contain features + target (output of add_features)
    - models must be fitted sklearn-like classifiers with predict_proba

    If date is None:
        picks a random row from the test segment (time-based split).
    If date is provided:
        must exist in the test segment.

    Returns a dict with per-model probs, average prob and final decision.
    """
    df_feat = df_feat.copy().reset_index(drop=True)

    # Basic checks
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df_feat.columns]
    if missing:
        raise ValueError(f"Missing columns in df_feat: {missing}")

    if "date" not in df_feat.columns:
        raise ValueError("df_feat must contain a 'date' column.")

    # Define test zone consistent with time split
    n = len(df_feat)
    cut = int(n * (1 - test_size))
    test_df = df_feat.iloc[cut:].copy()

    if len(test_df) == 0:
        raise ValueError("Test segment is empty. Check test_size or data length.")

    # Select one row
    if date is None:
        rng = np.random.default_rng(random_state)
        idx = int(rng.integers(0, len(test_df)))
        row = test_df.iloc[idx]
    else:
        target_date = pd.to_datetime(date)
        sub = test_df[test_df["date"] == target_date]
        if len(sub) == 0:
            raise ValueError("Date not found in the test segment.")
        row = sub.iloc[0]

    X_row = row[FEATURE_COLS].values.reshape(1, -1)

    # Per-model probabilities
    probas = {}
    for name, m in models.items():
        if not hasattr(m, "predict_proba"):
            raise ValueError(f"Model '{name}' does not support predict_proba.")
        probas[name] = float(m.predict_proba(X_row)[0, 1])

    avg_proba = float(np.mean(list(probas.values())))
    pred_label = int(avg_proba >= 0.5)

    return {
        "date": row["date"],
        "close_t": float(row["close"]) if "close" in row else None,
        "avg_proba_up": avg_proba,
        "predicted_direction": "UP" if pred_label == 1 else "DOWN",
        "model_probas_up": probas,
        "actual_y_up": int(row[TARGET_COL]),
    }

import pandas as pd
from src.features import add_features
from src.config import FEATURE_COLS, TARGET_COL


def _toy_df(n=30, with_vix=True):
    data = {
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": [1.0] * n,
        "high": [1.01] * n,
        "low": [0.99] * n,
        "close": [1.0, 1.01, 1.02, 1.01, 1.00, 1.02, 1.03, 1.04, 1.03, 1.05,
                  1.06, 1.07, 1.06, 1.05, 1.08, 1.09, 1.10, 1.09, 1.11, 1.12,
                  1.10, 1.09, 1.08, 1.10, 1.12, 1.13, 1.14, 1.15, 1.14, 1.16],
    }
    if with_vix:
        data["vix"] = [15.0] * n
    return pd.DataFrame(data)


def test_add_features_creates_all_columns_and_no_nan():
    df = _toy_df()
    out = add_features(df)
    for col in FEATURE_COLS + [TARGET_COL]:
        assert col in out.columns
    assert not out[FEATURE_COLS + [TARGET_COL]].isna().any().any()


def test_add_features_sets_target_direction():
    df = _toy_df()
    out = add_features(df)
    for i in range(3):  # vérifie les trois premières lignes après dropna
        row = out.iloc[i]
        assert row[TARGET_COL] == int(row["close_tomorrow"] > row["close"])


def test_add_features_handles_missing_vix_column():
    df = _toy_df(with_vix=False)
    out = add_features(df)
    # Si vix n'est pas présent en entrée, on tolère l'absence de vix_lag1.
    assert "vix_lag1" not in out.columns

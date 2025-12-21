# src/features.py
import pandas as pd
import numpy as np
from .config import FEATURE_COLS, TARGET_COL

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ret"] = df["close"].pct_change()

    for k in [1, 2, 3, 5]:
        df[f"ret_lag{k}"] = df["ret"].shift(k)

    df["ret_rollmean_5"] = df["ret"].rolling(5).mean()
    df["ret_rollstd_5"] = df["ret"].rolling(5).std()
    df["ret_rollstd_10"] = df["ret"].rolling(10).std()
    df["ret_rollstd_20"] = df["ret"].rolling(20).std()
    df["abs_ret_lag1"] = df["ret"].abs().shift(1)

    # NEW: EWMA vol (plus “finance” que rolling std)
    df["ret_ewmstd_20"] = df["ret"].ewm(span=20, adjust=False).std()

    # NEW: z-score ret (ret / vol récente)
    df["ret_z_20"] = df["ret"] / (df["ret_rollstd_20"] + 1e-12)

    if {"high", "low", "close"}.issubset(df.columns):
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    else:
        df["range_pct"] = np.nan

    if "vix" in df.columns:
        df["vix_lag1"] = df["vix"].shift(1)

    df["close_tomorrow"] = df["close"].shift(-1)
    df[TARGET_COL] = (df["close_tomorrow"] > df["close"]).astype(int)

    # IMPORTANT: dropna après TOUT
    subset = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df.dropna(subset=subset).reset_index(drop=True)

    return df

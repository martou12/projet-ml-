import pandas as pd
import numpy as np
from .config import FEATURE_COLS, TARGET_COL


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1) Return
    df["ret"] = df["close"].pct_change()

    # 2) Lags
    for k in [1, 2, 3, 5]:
        df[f"ret_lag{k}"] = df["ret"].shift(k)

    # 3) Rolling 5
    df["ret_rollmean_5"] = df["ret"].rolling(5).mean()
    df["ret_rollstd_5"] = df["ret"].rolling(5).std()

    # 4) Internal vol proxies
    df["ret_rollstd_10"] = df["ret"].rolling(10).std()
    df["ret_rollstd_20"] = df["ret"].rolling(20).std()
    df["abs_ret_lag1"] = df["ret"].abs().shift(1)

    # 5) OHLC range proxy
    if {"high", "low", "close"}.issubset(df.columns):
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    else:
        df["range_pct"] = np.nan

    # 6) External VIX lag if present
    if "vix" in df.columns:
        df["vix_lag1"] = df["vix"].shift(1)

    # 7) Target
    df["close_tomorrow"] = df["close"].shift(-1)
    df[TARGET_COL] = (df["close_tomorrow"] > df["close"]).astype(int)

    # 8) Dropna after ALL features exist
    subset = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df.dropna(subset=subset)

    return df

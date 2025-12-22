import pandas as pd
import numpy as np
from .config import FEATURE_COLS, TARGET_COL


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- FX basic ---
    df["ret"] = df["close"].pct_change()

    for k in [1, 2, 3, 5]:
        df[f"ret_lag{k}"] = df["ret"].shift(k)

    df["ret_rollmean_5"] = df["ret"].rolling(5).mean()
    df["ret_rollstd_5"] = df["ret"].rolling(5).std()

    # --- Volatility proxies (interne FX) ---
    df["ret_rollstd_10"] = df["ret"].rolling(10).std()
    df["ret_rollstd_20"] = df["ret"].rolling(20).std()
    df["abs_ret_lag1"] = df["ret"].abs().shift(1)

    if {"high", "low", "close"}.issubset(df.columns):
        df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    else:
        df["range_pct"] = np.nan

    # --- “plus propre” (indicateurs FX simples) ---
    df["mom_10"] = df["close"] / df["close"].shift(10) - 1
    df["ma20_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1
    df["rsi_14"] = _rsi(df["close"], 14)

    # MACD (un classique)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Jour de semaine en encoding cyclique (évite one-hot)
    if "date" in df.columns and np.issubdtype(df["date"].dtype, np.datetime64):
        dow = df["date"].dt.dayofweek  # 0..6
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # --- External series (si présentes) -> features safe (lag/returns) ---
    if "vix" in df.columns:
        df["vix_lag1"] = df["vix"].shift(1)

    if "dgs2" in df.columns:
        df["dgs2_lag1"] = df["dgs2"].shift(1)

    if "dgs10" in df.columns:
        df["dgs10_lag1"] = df["dgs10"].shift(1)

    if {"dgs10", "dgs2"}.issubset(df.columns):
        df["term_spread"] = df["dgs10"] - df["dgs2"]
        df["term_spread_lag1"] = df["term_spread"].shift(1)

    if "dtwexbgs" in df.columns:
        df["dtwexbgs_ret_lag1"] = df["dtwexbgs"].pct_change().shift(1)

    if "sp500" in df.columns:
        df["sp500_ret_lag1"] = df["sp500"].pct_change().shift(1)

    # --- Target J+1 (direction) ---
    df["close_tomorrow"] = df["close"].shift(-1)
    df[TARGET_COL] = (df["close_tomorrow"] > df["close"]).astype(int)

    # --- Drop NA after all features exist ---
    for c in [
        "vix_lag1",
        "dgs2_lag1", "dgs10_lag1", "term_spread_lag1",
        "dtwexbgs_ret_lag1",
        "sp500_ret_lag1",
    ]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # --- Drop NA only for columns that exist (utile si include_sp500=False etc.) ---
    subset = [c for c in FEATURE_COLS if c in df.columns] + [TARGET_COL]
    df = df.dropna(subset=subset).reset_index(drop=True)

    return df

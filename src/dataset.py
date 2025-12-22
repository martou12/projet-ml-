import pandas as pd
from .data_loading import load_eurusd
from .external_data import load_vix, load_dgs2, load_dgs10, load_dtwexbgs, load_sp500
from .features import add_features


def _merge_ffill(df: pd.DataFrame, ext: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.merge(ext, on="date", how="left").sort_values("date")
    df[col] = df[col].ffill()
    return df


def build_dataset(
    include_vix: bool = True,
    include_rates: bool = True,
    include_usd_index: bool = True,
    include_sp500: bool = True,
) -> pd.DataFrame:
    """
    Charge EURUSD puis merge des séries externes (ffill).
    Retourne df_feat = add_features(df) prêt pour modeling.
    """
    df = load_eurusd().sort_values("date")

    if include_vix:
        df = _merge_ffill(df, load_vix(), "vix")

    if include_rates:
        df = _merge_ffill(df, load_dgs2(), "dgs2")
        df = _merge_ffill(df, load_dgs10(), "dgs10")

    if include_usd_index:
        df = _merge_ffill(df, load_dtwexbgs(), "dtwexbgs")

    if include_sp500:
        df = _merge_ffill(df, load_sp500(), "sp500")

    df_feat = add_features(df).sort_values("date").reset_index(drop=True)
    return df_feat

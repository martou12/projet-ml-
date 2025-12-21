import pandas as pd
from .data_loading import load_eurusd
from .external_data import load_vix
from .features import add_features

def build_dataset(include_vix: bool = True) -> pd.DataFrame:
    df = load_eurusd()

    if include_vix:
        vix = load_vix()
        df = df.merge(vix, on="date", how="left").sort_values("date")
        df["vix"] = df["vix"].ffill()

    df_feat = add_features(df).sort_values("date").reset_index(drop=True)
    return df_feat

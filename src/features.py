import pandas as pd

# colonnes utilisées comme features dans le modèle
FEATURE_COLS = [
    "ret_lag1",
    "ret_lag2",
    "ret_lag3",
    "ret_lag5",
    "ret_rollmean_5",
    "ret_rollstd_5",
]

TARGET_COL = "y_up"  # 1 = hausse demain, 0 = baisse/égal


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features nécessaires et la cible y_up.
    On utilise uniquement des infos disponibles au temps t
    pour prédire la direction du close à t+1.
    """
    df = df.copy()

    # retour simple quotidien
    df["ret"] = df["close"].pct_change()

    # lags des retours (court terme)
    for k in [1, 2, 3, 5]:
        df[f"ret_lag{k}"] = df["ret"].shift(k)

    # moyenne et volatilité rolling sur 5 jours
    df["ret_rollmean_5"] = df["ret"].rolling(5).mean()
    df["ret_rollstd_5"] = df["ret"].rolling(5).std()

    # cible : 1 si close(t+1) > close(t), sinon 0
    df["close_tomorrow"] = df["close"].shift(-1)
    df[TARGET_COL] = (df["close_tomorrow"] > df["close"]).astype(int)

    # on enlève les lignes avec NaN (dus aux décalages/rolling)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    return df

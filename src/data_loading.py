import pandas as pd
from pathlib import Path
from .config import DATA_RAW


def load_eurusd(path: str | None = None) -> pd.DataFrame:
    """
    Charge les données EUR/USD (OHLC), nettoie un peu et trie par date.
    """
    # si aucun chemin passé → on prend le chemin par défaut défini dans config.py
    if path is None:
        path = DATA_RAW

    # lecture du fichier Excel
    df = pd.read_excel(path)

    # cas spécial : certains exports mettent tout dans une seule colonne CSV
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        header_parts = [p.strip() for p in str(df.columns[0]).split(",")]
        # on découpe la colonne en plusieurs colonnes en utilisant la virgule
        df = df.iloc[:, 0].str.split(",", expand=True)
        if len(header_parts) == df.shape[1]:
            # si on a autant de noms que de colonnes → on les utilise comme header
            df.columns = [h.lower() for h in header_parts]
        else:
            # sinon on met des noms génériques col0, col1, ...
            df.columns = [f"col{i}" for i in range(df.shape[1])]

    # tous les noms de colonnes en minuscules (plus simple à manipuler)
    df = df.rename(columns=str.lower)

    # conversion de la colonne date en vrai type datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # conversion des colonnes de prix en numériques
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # tri par date + remise à zéro de l’index
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    # on enlève les lignes où date ou close sont manquants
    if {"date", "close"}.issubset(df.columns):
        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
    # juste avant return df
    if "date" not in df.columns:
        raise ValueError("load_eurusd: colonne 'date' absente. Colonnes trouvées: " + str(df.columns.tolist()))

    df = df.drop_duplicates(subset=["date"]).reset_index(drop=True)

    # on renvoie le DataFrame propre
    return df

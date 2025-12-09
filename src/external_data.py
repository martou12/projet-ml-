import pandas as pd
from .config import PROJECT_ROOT

VIX_PATH = PROJECT_ROOT / "data" / "external" / "vix.csv"


def load_vix(path=VIX_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Nettoyage des noms de colonnes (espaces + BOM)
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # Normalisation simple
    rename_map = {}
    if "DATE" in df.columns:
        rename_map["DATE"] = "date"
    if "observation_date" in df.columns:
        rename_map["observation_date"] = "date"
    if "VIXCLS" in df.columns:
        rename_map["VIXCLS"] = "vix"

    df = df.rename(columns=rename_map)

    if "date" not in df.columns or "vix" not in df.columns:
        raise ValueError(
            f"load_vix: colonnes non reconnues dans {path}. "
            f"Colonnes trouvées: {df.columns.tolist()}"
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")

    return df.dropna(subset=["date", "vix"]).sort_values("date").reset_index(drop=True)

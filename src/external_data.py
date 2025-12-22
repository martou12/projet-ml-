import pandas as pd
from .config import PROJECT_ROOT

EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


def _load_observation_csv(path, value_name: str) -> pd.DataFrame:
    """
    Lit un CSV du type:
    observation_date,XXX
    2020-12-22,0.13
    """
    df = pd.read_csv(path)

    # clean headers
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

    # date column
    if "observation_date" in df.columns:
        df = df.rename(columns={"observation_date": "date"})
    elif "DATE" in df.columns:
        df = df.rename(columns={"DATE": "date"})

    if "date" not in df.columns:
        raise ValueError(f"Date column not found in {path}. Columns: {df.columns.tolist()}")

    # value column = first column that is not date
    value_cols = [c for c in df.columns if c != "date"]
    if len(value_cols) == 0:
        raise ValueError(f"No value column found in {path}. Columns: {df.columns.tolist()}")

    df = df.rename(columns={value_cols[0]: value_name})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")

    df = df.dropna(subset=["date", value_name]).sort_values("date").reset_index(drop=True)
    return df


def load_vix(path=EXTERNAL_DIR / "vix.csv") -> pd.DataFrame:
    return _load_observation_csv(path, "vix")


def load_dgs2(path=EXTERNAL_DIR / "DGS2.csv") -> pd.DataFrame:
    return _load_observation_csv(path, "dgs2")


def load_dgs10(path=EXTERNAL_DIR / "DGS10.csv") -> pd.DataFrame:
    return _load_observation_csv(path, "dgs10")


def load_dtwexbgs(path=EXTERNAL_DIR / "DTWEXBGS.csv") -> pd.DataFrame:
    return _load_observation_csv(path, "dtwexbgs")


def load_sp500(path=EXTERNAL_DIR / "SP500.csv") -> pd.DataFrame:
    return _load_observation_csv(path, "sp500")

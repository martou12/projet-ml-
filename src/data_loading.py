import pandas as pd
from pathlib import Path


def load_eurusd(path: str | None = None) -> pd.DataFrame:
    """
    Load EUR/USD OHLC data, handle CSV-in-XLSX edge case, clean columns, and sort by date.
    """
    if path is None:
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / "data" / "eurusd_d.xlsx"

    df = pd.read_excel(path)

    # Some exports are a CSV packed into a single Excel column; split if detected.
    if len(df.columns) == 1 and "," in str(df.columns[0]):
        header_parts = [p.strip() for p in str(df.columns[0]).split(",")]
        df = df.iloc[:, 0].str.split(",", expand=True)
        if len(header_parts) == df.shape[1]:
            df.columns = [h.lower() for h in header_parts]
        else:
            df.columns = [f"col{i}" for i in range(df.shape[1])]

    # Normalize column names
    df = df.rename(columns=str.lower)

    # Parse dates and numeric prices
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort and clean
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    if {"date", "close"}.issubset(df.columns):
        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)

    return df

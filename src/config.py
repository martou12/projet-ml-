from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "eurusd_d.xlsx"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "eurusd_features.csv"

FEATURE_COLS = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "ret_rollmean_5", "ret_rollstd_5",
    "ret_rollstd_10", "ret_rollstd_20","abs_ret_lag1",
    "range_pct","vix_lag1"

]
TARGET_COL = "y_up"
TEST_SIZE = 0.2

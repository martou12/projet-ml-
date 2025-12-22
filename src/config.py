from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "eurusd_d.xlsx"

TARGET_COL = "y_up"
TEST_SIZE = 0.2

FEATURE_COLS = [
    # base returns
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "ret_rollmean_5", "ret_rollstd_5",

    # vol proxies
    "ret_rollstd_10", "ret_rollstd_20",
    "abs_ret_lag1",
    "range_pct",

    # “clean FX”
    "mom_10", "ma20_ratio", "rsi_14",
    "macd_hist",
    "dow_sin", "dow_cos",

    # external
    "vix_lag1",
    "dgs2_lag1", "dgs10_lag1", "term_spread_lag1",
    "dtwexbgs_ret_lag1",
    "sp500_ret_lag1",
]

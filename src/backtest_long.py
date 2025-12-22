import pandas as pd
from src.backtest import backtest_pnl


def main():
    res = backtest_pnl(
        n_splits=5,
        threshold=0.5,
        include_vix=True,
        capital_init=10_000,
        model_type="voting_xgb",
        position_mode="long_only",
        position_size=1.0,
        fee_per_trade=0.0,
        start_date="2004-01-01",
        end_date="2024-01-01",
    )
    print(pd.DataFrame([res["summary"]]))


if __name__ == "__main__":
    main()

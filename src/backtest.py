import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone

from src.dataset import build_dataset
from src.modeling import build_logreg_pipeline
from src.config import FEATURE_COLS, TARGET_COL


def backtest_pnl(
    n_splits: int = 5,
    threshold: float = 0.5,
    # Feature flags (must match what build_dataset can provide)
    include_vix: bool = True,
    include_rates: bool = True,
    include_usd_index: bool = True,
    include_sp500: bool = True,
    # Trading parameters
    capital_init: float = 10_000.0,
    position_mode: str = "long_only",  # "long_only", "long_short", "short_only"
    position_size: float = 1.0,        # fraction of capital exposed (scales returns)
    fee_per_trade: float = 0.0,        # fraction cost paid when position changes
    start_date: str | None = None,
    end_date: str | None = None,
    model=None,
):
    """
    Walk-forward backtest using TimeSeriesSplit:
    - Trains on past data, predicts on the next block (no shuffling).
    - gap=1 to avoid leakage when your target is "tomorrow direction" (t+1).
    - Signal is derived from predicted probability >= threshold.
    - PnL uses position(t) applied to market return(t+1) via a shift.
    - Fees are paid when the position changes.
    """

    # ------------------------------------------------------------------
    # 1) Build the full dataset (features + target), sorted by date
    # ------------------------------------------------------------------
    df = (
        build_dataset(
            include_vix=include_vix,
            include_rates=include_rates,
            include_usd_index=include_usd_index,
            include_sp500=include_sp500,
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # 2) Optional date filtering
    # ------------------------------------------------------------------
    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)

    if len(df) < 200:
        raise ValueError("Not enough data for a robust backtest (dataset too short).")

    # ------------------------------------------------------------------
    # 3) Market returns (close-to-close). Used as underlying PnL driver.
    # ------------------------------------------------------------------
    df["ret_mkt"] = df["close"].pct_change().fillna(0.0)

    # ------------------------------------------------------------------
    # 4) ML matrices
    # ------------------------------------------------------------------
    X = df[FEATURE_COLS].to_numpy()
    y = df[TARGET_COL].to_numpy()
    dates = df["date"]

    # gap=1 prevents training on points too close to the test (target horizon is t+1)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)

    # Store out-of-sample predictions aligned on the original index
    proba_all = pd.Series(index=df.index, dtype=float)
    pred_all = pd.Series(index=df.index, dtype=float)
    folds_info: list[dict] = []

    # ------------------------------------------------------------------
    # 5) Walk-forward training / prediction
    # ------------------------------------------------------------------
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Build a fresh model each fold (avoid state leakage across folds)
        if model is None:
            model_i = build_logreg_pipeline()
        else:
            model_i = model() if callable(model) else clone(model)

        model_i.fit(X_train, y_train)

        proba = model_i.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        proba_all.iloc[test_idx] = proba
        pred_all.iloc[test_idx] = preds

        folds_info.append(
            {
                "fold": fold,
                "train_start": str(dates.iloc[train_idx[0]].date()),
                "train_end": str(dates.iloc[train_idx[-1]].date()),
                "test_start": str(dates.iloc[test_idx[0]].date()),
                "test_end": str(dates.iloc[test_idx[-1]].date()),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

    # ------------------------------------------------------------------
    # 6) Convert predictions into positions (only where we have predictions)
    # ------------------------------------------------------------------
    mask = pred_all.notna()
    pos = pd.Series(0.0, index=df.index)

    if position_mode == "long_only":
        # 1 => long, 0 => flat
        pos.loc[mask] = pred_all.loc[mask].astype(int)

    elif position_mode == "long_short":
        # 1 => long, 0 => short (-1)
        pos.loc[mask] = pred_all.loc[mask].replace({0: -1, 1: 1}).astype(int)

    elif position_mode == "short_only":
        # 0 => short (-1), 1 => flat (0)
        pos.loc[mask] = pred_all.loc[mask].replace({1: 0, 0: -1}).astype(int)

    else:
        raise ValueError("position_mode must be 'long_only', 'long_short', or 'short_only'.")

    # Scale exposure (e.g., 0.5 means half exposure, 2.0 means leveraged)
    pos = pos * float(position_size)

    # ------------------------------------------------------------------
    # 7) Alignment: position decided at t is applied to return at t+1
    #    => shift positions by 1 bar.
    # ------------------------------------------------------------------
    pos_applied = pos.shift(1).fillna(0.0)

    # ------------------------------------------------------------------
    # 8) Transaction fees: pay a fee whenever position changes.
    #    The change happens at time t, and we subtract it from the next applied step (t+1),
    #    to match the shifted position logic.
    # ------------------------------------------------------------------
    trade_flag = (pos != pos.shift(1)).astype(int).fillna(0)
    fees = trade_flag.shift(1).fillna(0.0) * float(fee_per_trade)

    # ------------------------------------------------------------------
    # 9) Strategy returns and equity curves
    # ------------------------------------------------------------------
    strat_ret = pos_applied * df["ret_mkt"] - fees
    equity = capital_init * (1.0 + strat_ret).cumprod()

    # Benchmark: buy & hold (always long on EURUSD return series)
    equity_mkt = capital_init * (1.0 + df["ret_mkt"]).cumprod()

    # ------------------------------------------------------------------
    # 10) Basic performance stats
    # ------------------------------------------------------------------
    max_dd = float(((equity / equity.cummax()) - 1.0).min())
    trade_rate = float((pos != 0.0).mean())
    entry_flag = ((pos != 0.0) & (pos.shift(1).fillna(0.0) == 0.0)).astype(int)

    mu = float(strat_ret.mean())
    sig = float(strat_ret.std())
    sharpe = (mu / sig * np.sqrt(252)) if sig > 1e-12 else 0.0

    summary = {
        "start": str(dates.iloc[0].date()),
        "end": str(dates.iloc[-1].date()),
        "capital_init": float(capital_init),
        "capital_final": float(equity.iloc[-1]),
        "pnl_eur": float(equity.iloc[-1] - capital_init),
        "pnl_pct": float((equity.iloc[-1] / capital_init - 1.0) * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "trade_rate_nonzero": trade_rate,
        "n_trades": int(entry_flag.sum()),
        "sharpe_annualized": float(sharpe),
    }

    # ------------------------------------------------------------------
    # 11) Outputs
    # ------------------------------------------------------------------
    signals = pd.DataFrame(
        {
            "date": dates,
            "ret_mkt": df["ret_mkt"],
            "proba_up": proba_all,
            "pred": pred_all,
            "position": pos_applied,
            "fees": fees,
            "strat_ret": strat_ret,
            "equity": equity,
            "equity_mkt": equity_mkt,
        }
    )

    return {
        "folds": pd.DataFrame(folds_info),
        "summary": summary,
        "equity_curve": equity,
        "equity_curve_mkt": equity_mkt,
        "signals": signals,
    }

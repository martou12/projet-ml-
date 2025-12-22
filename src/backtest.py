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
    # ✅ on aligne avec ton dataset.py (toutes les datas externes)
    include_vix: bool = True,
    include_rates: bool = True,
    include_usd_index: bool = True,
    include_sp500: bool = True,
    # trading params
    capital_init: float = 10_000.0,
    position_mode: str = "long_only",  # "long_only", "long_short", "short_only"
    position_size: float = 1.0,        # fraction du capital exposée
    fee_per_trade: float = 0.0,        # coût (fraction) payé quand on change de position
    start_date: str | None = None,
    end_date: str | None = None,
    model=None,
):
    """
    Backtest simple walk-forward:
    - TimeSeriesSplit (gap=1) pour éviter la fuite (horizon J+1)
    - Signaux via Logistic Regression (pipeline du projet)
    - PnL: position(t) appliquée sur ret(t+1) (via shift)
    - frais payés quand on change de position
    """

    # 1) dataset complet (déjà features + target)
    df = build_dataset(
        include_vix=include_vix,
        include_rates=include_rates,
        include_usd_index=include_usd_index,
        include_sp500=include_sp500,
    ).sort_values("date").reset_index(drop=True)

    # 2) filtre période
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)

    if len(df) < 200:
        raise ValueError("Pas assez de données pour faire un backtest propre (dataset trop court).")

    # 3) ret marché (close-to-close)
    df["ret_mkt"] = df["close"].pct_change().fillna(0.0)

    # 4) matrices ML
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    dates = df["date"]

    # gap=1 pour horizon J+1
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)

    proba_all = pd.Series(index=df.index, dtype=float)
    pred_all = pd.Series(index=df.index, dtype=float)
    folds = []

    # 5) walk-forward
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        if model is None:
            model_i = build_logreg_pipeline()
        else:
            if callable(model):
                model_i = model()
            else:
                model_i = clone(model)

        model_i.fit(X_train, y_train)

        proba = model_i.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        proba_all.iloc[test_idx] = proba
        pred_all.iloc[test_idx] = preds

        folds.append({
            "fold": fold,
            "train_start": str(dates.iloc[train_idx[0]].date()),
            "train_end": str(dates.iloc[train_idx[-1]].date()),
            "test_start": str(dates.iloc[test_idx[0]].date()),
            "test_end": str(dates.iloc[test_idx[-1]].date()),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    # 6) positions (⚠️ on garde NaN => 0, pas short par défaut)
    pos = pd.Series(0.0, index=df.index)
    mask = pred_all.notna()

    if position_mode == "long_only":
        # 1 ou 0 uniquement là où on a une prédiction
        pos.loc[mask] = pred_all.loc[mask].astype(int)

    elif position_mode == "long_short":
        # 1 ou -1 uniquement là où on a une prédiction
        pos.loc[mask] = pred_all.loc[mask].replace({0: -1, 1: 1}).astype(int)

    elif position_mode == "short_only":
        # 0 ou -1 uniquement là où on a une prédiction
        pos.loc[mask] = pred_all.loc[mask].replace({1: 0, 0: -1}).astype(int)

    else:
        raise ValueError("position_mode doit être 'long_only', 'long_short' ou 'short_only'.")

    # exposition
    pos = pos * float(position_size)

    # 7) alignement: position(t) -> perf sur ret(t+1) => on applique position shiftée sur ret_mkt
    pos_applied = pos.shift(1).fillna(0.0)

    # 8) frais: on paie quand on change la position (à t), donc on soustrait au moment où on applique (t+1)
    trade_flag = (pos != pos.shift(1)).astype(int).fillna(0)
    fees = trade_flag.shift(1).fillna(0.0) * float(fee_per_trade)

    # 9) stratégie
    strat_ret = pos_applied * df["ret_mkt"] - fees
    equity = capital_init * (1 + strat_ret).cumprod()

    # buy&hold pour comparer (juste long EURUSD)
    equity_mkt = capital_init * (1 + df["ret_mkt"]).cumprod()

    # 10) stats simples
    max_dd = float(((equity / equity.cummax()) - 1).min())
    trade_rate = float((pos != 0).mean())

    # sharpe daily approx (si std=0 => 0)
    mu = float(strat_ret.mean())
    sig = float(strat_ret.std())
    sharpe = (mu / sig * np.sqrt(252)) if sig > 1e-12 else 0.0

    summary = {
        "start": str(dates.iloc[0].date()),
        "end": str(dates.iloc[-1].date()),
        "capital_init": float(capital_init),
        "capital_final": float(equity.iloc[-1]),
        "pnl_eur": float(equity.iloc[-1] - capital_init),
        "pnl_pct": float((equity.iloc[-1] / capital_init - 1) * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "trade_rate_nonzero": trade_rate,
        "n_trades": int(trade_flag.sum()),
        "sharpe_annualized": float(sharpe),
    }

    return {
        "folds": pd.DataFrame(folds),
        "summary": summary,
        "equity_curve": equity,
        "equity_curve_mkt": equity_mkt,
        "signals": pd.DataFrame({
            "date": dates,
            "ret_mkt": df["ret_mkt"],
            "proba_up": proba_all,
            "pred": pred_all,
            "position": pos_applied,
            "fees": fees,
            "strat_ret": strat_ret,
            "equity": equity,
            "equity_mkt": equity_mkt,
        }),
    }

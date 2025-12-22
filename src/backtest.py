import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.dataset import build_dataset
from src.modeling import build_logreg_pipeline
from src.config import FEATURE_COLS, TARGET_COL


def backtest_pnl(
    n_splits: int = 5,
    threshold: float = 0.5,
    include_vix: bool = True,
    capital_init: float = 10_000.0,
    position_mode: str = "long_only",  # "long_only" (1/0), "long_short" (1/-1), "short_only" (0/-1)
    position_size: float = 1.0,        # fraction du capital engagée (1.0 = 100 %, 0.5 = 50 %)
    fee_per_trade: float = 0.0,        # coût proportionnel par changement de position (ex. 0.0001 ~ 1 pip)
    start_date: str | None = None,     # ex. "2004-01-01" pour filtrer la période
    end_date: str | None = None,       # ex. "2024-01-01"
):
    """
    Backtest simple en walk-forward :
    - split temporel avec TimeSeriesSplit (gap=1 pour éviter fuite)
    - signaux issus de build_logreg_pipeline
    - P&L calculé en appliquant la position décalée d'un jour sur le retour du lendemain
    - fees appliqués lors des changements de position (turnover)
    """
    df = build_dataset(include_vix=include_vix).sort_values("date")
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)
    if "close" not in df.columns:
        raise ValueError("La colonne 'close' est requise pour le P&L.")

    # Rendement spot jour/jour
    df["ret"] = df["close"].pct_change()

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    dates = df["date"]

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)

    # Pour stocker les probabilités/prédictions alignées au test
    proba_all = pd.Series(index=df.index, dtype=float)
    pred_all = pd.Series(index=df.index, dtype=float)
    folds = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        model = build_logreg_pipeline()
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
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

    # Construction de la position quotidienne
    if position_mode == "long_only":
        position = pred_all.fillna(0)  # 1 ou 0
    elif position_mode == "long_short":
        position = pred_all.fillna(0).replace({0: -1})  # 1 ou -1
    elif position_mode == "short_only":
        position = pred_all.fillna(0).replace({1: 0, 0: -1})  # 0 ou -1
    else:
        raise ValueError("position_mode doit être 'long_only', 'long_short' ou 'short_only'.")

    # Applique un pourcentage de capital (ex. 0.5 pour 50 %)
    position = position * position_size

    # Décalage : position du jour t appliquée au retour du jour t+1 (évite fuite)
    position_shifted = position.shift(1).fillna(0)

    # Turnover = changements de position -> frais
    turnover = position_shifted.diff().abs().fillna(0)
    fees = turnover * fee_per_trade

    strat_ret = position_shifted * df["ret"] - fees
    equity = capital_init * (1 + strat_ret).cumprod()
    liquidated = bool((equity <= 0).any())

    summary = {
        "start": str(dates.iloc[0].date()),
        "end": str(dates.iloc[-1].date()),
        "capital_init": float(capital_init),
        "capital_final": float(equity.iloc[-1]),
        "pnl_eur": float(equity.iloc[-1] - capital_init),
        "pnl_pct": float((equity.iloc[-1] / capital_init - 1) * 100),
        "max_drawdown_pct": float(((equity / equity.cummax()) - 1).min() * 100),
        "liquidated": liquidated,
    }

    return {
        "folds": pd.DataFrame(folds),
        "summary": summary,
        "equity_curve": equity,
        "signals": pd.DataFrame({
            "date": dates,
            "ret": df["ret"],
            "proba_up": proba_all,
            "position": position_shifted,
            "strat_ret": strat_ret,
            "equity": equity,
            "fees": fees,
        }),
    }


if __name__ == "__main__":
    res = backtest_pnl(
        n_splits=5,
        threshold=0.5,
        include_vix=True,
        capital_init=10_000,
        position_mode="long_only",
        position_size=1.0,        # 1.0 = 100 % du capital, 0.5 = 50 %, etc.
        fee_per_trade=0.0,        # ajoute un coût par changement de position si souhaité
        start_date="2004-01-01",  # borne basse pour le backtest
        end_date="2024-01-01",    # borne haute pour le backtest
    )
    print(pd.DataFrame([res["summary"]]))

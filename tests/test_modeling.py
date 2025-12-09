import pandas as pd
from src.features import add_features
from src.modeling import train_test_split_time
from src.config import TEST_SIZE


def _toy_feat_df(n=40):
    base = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "open": range(n),
        "high": [x + 0.5 for x in range(n)],
        "low": [x - 0.5 for x in range(n)],
        "close": [1.0 + 0.01 * x for x in range(n)],
        "vix": [15.0] * n,
    })
    return add_features(base)


def test_train_test_split_time_respects_order_and_size():
    df_feat = _toy_feat_df()
    X_train, X_test, y_train, y_test = train_test_split_time(df_feat, test_size=TEST_SIZE)
    expected_test_len = int(len(df_feat) * TEST_SIZE)
    assert len(X_test) == expected_test_len
    assert len(y_test) == expected_test_len
    train_last_date = df_feat.iloc[len(X_train) - 1]["date"]
    test_first_date = df_feat.iloc[len(X_train)]["date"]
    assert train_last_date < test_first_date


def test_train_test_split_time_on_small_sample():
    df_feat = _toy_feat_df(n=25)
    X_train, X_test, y_train, y_test = train_test_split_time(df_feat, test_size=0.25)
    assert len(X_train) > 0
    assert len(X_test) > 0

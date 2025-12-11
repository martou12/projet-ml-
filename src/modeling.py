import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from .config import FEATURE_COLS, TARGET_COL, TEST_SIZE


def train_test_split_time(df, test_size: float = TEST_SIZE):
    """
    Split temporel simple : 80% train, 20% test (par défaut).
    Pas de shuffle.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    n = len(df)
    cut = int(n * (1 - test_size))

    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]

    return X_train, X_test, y_train, y_test


def build_logreg_pipeline():
    """
    Baseline : StandardScaler + LogisticRegression.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    return pipe


def build_decision_tree():
    
    return DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=50,
        random_state=0,
    )


def build_random_forest():
   
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        random_state=0,
        n_jobs=-1,
    )


def build_gradient_boosting():
    """
    Modèle Gradient Boosting (standard).
    """
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=0,
    )


def grid_search_timeseries(model, param_grid, X_train, y_train, n_splits: int = 5):
    """
    Petit GridSearch avec TimeSeriesSplit pour éviter la fuite d'information.
    On optimise la ROC-AUC.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    best_params = gs.best_params_
    best_score = gs.best_score_
    return best_model, best_params, best_score

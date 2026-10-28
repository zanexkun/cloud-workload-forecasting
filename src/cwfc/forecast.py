"""Forecasting methods. Each one takes the training series and returns n forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)
ROLL = 12


def last_value(train: pd.Series, n: int) -> np.ndarray:
    """Repeat the last observed value."""
    return np.repeat(float(train.iloc[-1]), n)


def train_mean(train: pd.Series, n: int) -> np.ndarray:
    """Repeat the mean of the whole training window."""
    return np.repeat(float(train.mean()), n)


def moving_avg(train: pd.Series, n: int, window: int = ROLL) -> np.ndarray:
    """Repeat the mean of the last `window` values (one hour at 5 minute cadence)."""
    return np.repeat(float(train.iloc[-window:].mean()), n)


def blend(train: pd.Series, n: int) -> np.ndarray:
    """Final method: average of last value and training mean.

    Starts from the current level but expects a pull back toward the normal level.
    """
    return (last_value(train, n) + train_mean(train, n)) / 2


def make_features(s: pd.Series) -> pd.DataFrame:
    """Lag features built only from past values of the series."""
    df = pd.DataFrame({"y": np.asarray(s, dtype=float)})
    for lag in LAGS:
        df[f"lag{lag}"] = df["y"].shift(lag)
    df[f"roll{ROLL}"] = df["y"].shift(1).rolling(ROLL).mean()
    return df.dropna()


def ridge_lags(train: pd.Series, n: int, alpha: float = 1.0) -> np.ndarray:
    """Ridge regression on lag features, forecasting recursively n steps ahead.

    Falls back to the last value when the series is too short to build features.
    """
    from sklearn.linear_model import Ridge

    feats = make_features(train)
    if len(feats) < 10:
        return last_value(train, n)
    X, y = feats.drop(columns="y"), feats["y"]
    model = Ridge(alpha=alpha).fit(X, y)
    history = list(np.asarray(train, dtype=float))
    preds = []
    for _ in range(n):
        h = pd.Series(history)
        row = {f"lag{lag}": h.iloc[-lag] for lag in LAGS}
        row[f"roll{ROLL}"] = h.iloc[-ROLL:].mean()
        p = float(model.predict(pd.DataFrame([row])[X.columns])[0])
        preds.append(p)
        history.append(p)
    return np.array(preds)


METHODS = {
    "last value": last_value,
    "train mean": train_mean,
    "moving avg 1h": moving_avg,
    "ridge lags": ridge_lags,
    "blend": blend,
}
FINAL_METHOD = "blend"

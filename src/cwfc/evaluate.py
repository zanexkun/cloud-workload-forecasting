"""Reproduce the method comparison, the holdout scores and the safety margin.

Usage:
    python -m cwfc.evaluate --data challenge_data

Validation only uses the training window (rolling origin, expanding window).
The holdout is scored once at the end, after the final method was fixed.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cwfc.data import build_targets, load_cpu, split_holdout
from cwfc.forecast import METHODS, FINAL_METHOD
from cwfc.metrics import mae, rmse


def rolling_origins(n_train: int, horizon: int = 24, step: int = 12, min_train: int = 50) -> list[int]:
    """Origins that end the last validation window exactly at the end of training."""
    last = n_train - horizon
    origins = list(range(last, min_train - 1, -step))[::-1]
    return origins if origins else ([last] if last > 0 else [])


def score(real, pred) -> dict:
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {"mae": mae(real, pred), "rmse": rmse(real, pred), "bias": float((pred - real).mean())}


def validate(series: pd.Series, horizon: int = 24) -> tuple[pd.DataFrame, np.ndarray]:
    rows, final_misses = [], []
    for origin in rolling_origins(len(series), horizon):
        train, test = series.iloc[:origin], series.iloc[origin:origin + horizon]
        for name, fn in METHODS.items():
            pred = fn(train, len(test))
            rows.append({"origin": origin, "method": name, **score(test, pred)})
            if name == FINAL_METHOD:
                final_misses.extend(test.to_numpy() - pred)
    return pd.DataFrame(rows), np.array(final_misses)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="validation, holdout scores and safety margin")
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args(argv)

    targets = build_targets(load_cpu(args.data))
    train, holdout = split_holdout(targets, args.horizon)
    y_train = train["aggregate_cpu_utilization"].reset_index(drop=True)
    y_hold = holdout["aggregate_cpu_utilization"].to_numpy()

    results, misses = validate(y_train, args.horizon)
    if results.empty:
        print(f"training window too short for validation ({len(y_train)} points), skipping it")
    else:
        print("validation (rolling origin inside the training window), mean over origins:")
        print(results.groupby("method")[["mae", "rmse", "bias"]].mean().sort_values("mae").round(3))

    print(f"\nholdout scores (final method fixed in advance: {FINAL_METHOD}):")
    hold = pd.DataFrame([{"method": name, **score(y_hold, fn(y_train, args.horizon))}
                         for name, fn in METHODS.items()]).set_index("method").sort_values("mae")
    print(hold.round(3))
    print("normalized mae (final / last value):",
          round(hold.loc[FINAL_METHOD, "mae"] / hold.loc["last value", "mae"], 3))

    if len(misses):
        final_pred = METHODS[FINAL_METHOD](y_train, args.horizon)
        print("\nsafety margin from validation misses (real minus forecast):")
        for level in (0.80, 0.90, 0.95, 1.00):
            margin = float(np.quantile(misses, level))
            covered = 100 * float((y_hold <= final_pred + margin).mean())
            print(f"  {int(level * 100)}%: +{margin:.3f} points "
                  f"({100 * margin / final_pred[0]:.1f}% of forecast), holdout covered {covered:.1f}%")


if __name__ == "__main__":
    main()

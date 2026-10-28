"""Generate predictions.csv for the final holdout.

Usage:
    python -m cwfc.predict --data challenge_data --output predictions.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data import build_targets, load_cpu, split_holdout
from cwfc.forecast import METHODS, FINAL_METHOD, last_value


def make_predictions(data_dir: str | Path, horizon: int = 24) -> pd.DataFrame:
    targets = build_targets(load_cpu(data_dir))
    train, holdout = split_holdout(targets, horizon)
    # only training values are used below, the holdout targets are never read
    vm_pred = last_value(train["active_vm_count"], horizon)
    cpu_pred = METHODS[FINAL_METHOD](train["aggregate_cpu_utilization"], horizon)
    return pd.DataFrame({
        "timestamp": holdout["timestamp"].to_numpy(),
        "active_vm_count": np.clip(np.round(vm_pred), 0, None).astype(int),
        "aggregate_cpu_utilization": np.clip(cpu_pred, 0, 100),
    })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="forecast the final holdout and write predictions.csv")
    parser.add_argument("--data", default="challenge_data", help="folder with vm_cpu_readings.csv.gz")
    parser.add_argument("--output", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--horizon", type=int, default=24, help="number of final timestamps to forecast")
    args = parser.parse_args(argv)

    preds = make_predictions(args.data, args.horizon)
    preds.to_csv(args.output, index=False)
    print(f"wrote {len(preds)} rows to {args.output}")


if __name__ == "__main__":
    main()

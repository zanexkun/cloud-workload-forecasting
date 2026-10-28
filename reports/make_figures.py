"""Rebuild the report figures in reports/figures.

Usage (from the repository root):
    python reports/make_figures.py --data challenge_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cwfc.data import build_targets, load_cpu, split_holdout
from cwfc.evaluate import validate
from cwfc.forecast import METHODS

OUT = Path(__file__).parent / "figures"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    targets = build_targets(load_cpu(args.data))
    train, hold = split_holdout(targets, args.horizon)
    y = train["aggregate_cpu_utilization"].reset_index(drop=True)
    h = hold["aggregate_cpu_utilization"].to_numpy()
    hrs_tr = train["timestamp"].reset_index(drop=True) / 3600
    hrs_h = hold["timestamp"] / 3600
    split_h = hold["timestamp"].iloc[0] / 3600
    plt.rcParams.update({"font.size": 11})

    # figure 1: training window only
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hrs_tr, y, color="#2b6cb0")
    ax.axhline(y.mean(), color="gray", ls=":", label=f"training mean {y.mean():.2f}%")
    ax.set_xlabel("hours")
    ax.set_ylabel("fleet average cpu (%)")
    ax.set_title(f"training window (first {len(y)} points)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig1_training.png", dpi=130)

    # figure 2: holdout with forecast, baseline and 95% safety margin
    results, misses = validate(y, args.horizon)
    blend = METHODS["blend"](y, args.horizon)
    last = METHODS["last value"](y, args.horizon)
    margin = float(np.quantile(misses, 0.95)) if len(misses) else 0.0
    start = max(0, len(y) - 50)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(hrs_tr.iloc[start:], y.iloc[start:], color="black", label="real (training)")
    ax.plot(hrs_h, h, color="black", alpha=0.6, label="real (holdout)")
    ax.plot(hrs_h, blend, color="green", label="blend forecast")
    ax.plot(hrs_h, last, color="orange", ls="--", label="last value baseline")
    ax.fill_between(hrs_h, blend, blend + margin, color="blue", alpha=0.12, label="95% safety margin")
    ax.axvline(split_h, color="red", lw=1, label="split")
    ax.set_xlabel("hours")
    ax.set_ylabel("fleet average cpu (%)")
    ax.set_title("holdout: forecast, baseline and safety margin")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_holdout.png", dpi=130)

    # figure 3: validation error per round
    if not results.empty:
        order = ["blend", "last value", "train mean", "ridge lags", "moving avg 1h"]
        p = results.pivot(index="origin", columns="method", values="mae")[order]
        fig, ax = plt.subplots(figsize=(10, 4.5))
        p.plot(kind="bar", ax=ax, width=0.8, rot=0)
        ax.set_ylim(0, p.to_numpy().max() * 1.2)
        ax.set_xlabel("validation round (origin point)")
        ax.set_ylabel("mae (points)")
        ax.set_title("validation error per round", pad=28)
        ax.legend(fontsize=9, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.tight_layout()
        fig.savefig(OUT / "fig3_validation.png", dpi=130)

    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()

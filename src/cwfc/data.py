"""Load the VM CPU readings and build the two fleet level target series."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CPU_FILE = "vm_cpu_readings.csv.gz"
REQUIRED_COLUMNS = {"timestamp", "vm_id", "avg_cpu"}


def load_cpu(data_dir: str | Path) -> pd.DataFrame:
    """Read the CPU readings file from the data directory."""
    path = Path(data_dir) / CPU_FILE
    if not path.exists():
        raise FileNotFoundError(f"could not find {path}")
    cpu = pd.read_csv(path, usecols=lambda c: c in REQUIRED_COLUMNS)
    missing = REQUIRED_COLUMNS - set(cpu.columns)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    return cpu


def build_targets(cpu: pd.DataFrame) -> pd.DataFrame:
    """Build one row per timestamp with both targets.

    active_vm_count: distinct VMs with a valid avg_cpu reading at that timestamp,
    so a VM is counted at most once even if a reading is duplicated.
    aggregate_cpu_utilization: mean of the VM level avg_cpu readings at that
    timestamp, each VM counted once, clipped to [0, 100].
    Rows with a missing avg_cpu are ignored, no values are invented.
    """
    valid = cpu.dropna(subset=["timestamp", "vm_id", "avg_cpu"])
    # one reading per VM per timestamp (average duplicates if any exist)
    per_vm = valid.groupby(["timestamp", "vm_id"], as_index=False)["avg_cpu"].mean()
    targets = per_vm.groupby("timestamp").agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
    )
    targets["aggregate_cpu_utilization"] = targets["aggregate_cpu_utilization"].clip(0, 100)
    return targets.sort_index().reset_index()


def split_holdout(targets: pd.DataFrame, horizon: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: everything except the last `horizon` timestamps is training."""
    if len(targets) <= horizon:
        raise ValueError(f"need more than {horizon} timestamps, got {len(targets)}")
    targets = targets.sort_values("timestamp").reset_index(drop=True)
    return targets.iloc[:-horizon].copy(), targets.iloc[-horizon:].copy()

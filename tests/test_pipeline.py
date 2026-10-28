import numpy as np
import pandas as pd
import pytest

from cwfc.data import build_targets, split_holdout
from cwfc.evaluate import rolling_origins
from cwfc.forecast import blend, last_value, make_features, train_mean
from cwfc.predict import make_predictions


def write_smoke_data(folder, n_timestamps=40, n_vms=5, start=1000, step=300):
    rng = np.random.default_rng(0)
    rows = []
    for t in range(n_timestamps):
        for v in range(n_vms):
            avg = float(rng.uniform(1, 30))
            rows.append({"timestamp": start + t * step, "vm_id": f"vm{v}",
                         "min_cpu": avg / 2, "max_cpu": avg * 2, "avg_cpu": avg})
    pd.DataFrame(rows).to_csv(folder / "vm_cpu_readings.csv.gz", index=False)
    return start, step


def test_targets_count_each_vm_once_and_average():
    cpu = pd.DataFrame({
        "timestamp": [0, 0, 0, 300, 300],
        "vm_id": ["a", "a", "b", "a", "b"],   # vm a appears twice at t=0
        "avg_cpu": [10.0, 10.0, 30.0, 5.0, np.nan],
    })
    t = build_targets(cpu)
    assert list(t["timestamp"]) == [0, 300]
    assert list(t["active_vm_count"]) == [2, 1]          # duplicate counted once, missing reading ignored
    assert t["aggregate_cpu_utilization"].tolist() == [20.0, 5.0]


def test_split_is_chronological():
    t = pd.DataFrame({"timestamp": [600, 0, 300, 900], "active_vm_count": 1,
                      "aggregate_cpu_utilization": 1.0})
    train, hold = split_holdout(t, horizon=2)
    assert list(train["timestamp"]) == [0, 300]
    assert list(hold["timestamp"]) == [600, 900]
    with pytest.raises(ValueError):
        split_holdout(t, horizon=4)


def test_baselines_and_blend():
    s = pd.Series([8.0, 10.0, 12.0])
    assert np.allclose(last_value(s, 3), 12.0)
    assert np.allclose(train_mean(s, 3), 10.0)
    assert np.allclose(blend(s, 3), 11.0)


def test_lag_features_use_only_the_past():
    s = pd.Series(np.arange(30, dtype=float))
    f = make_features(s)
    assert (f["lag1"] == f["y"] - 1).all()
    assert (f["roll12"] < f["y"]).all()   # the rolling mean never includes the current value


def test_rolling_origins_stay_inside_training():
    origins = rolling_origins(110, horizon=24)
    assert origins == [50, 62, 74, 86]
    assert origins[-1] + 24 == 110


def test_evaluate_runs_on_tiny_data(tmp_path, capsys):
    from cwfc.evaluate import main
    write_smoke_data(tmp_path, n_timestamps=30)
    main(["--data", str(tmp_path)])
    out = capsys.readouterr().out
    assert "holdout scores" in out


def test_predict_on_smoke_dataset(tmp_path):
    start, step = write_smoke_data(tmp_path, n_timestamps=40)
    preds = make_predictions(tmp_path, horizon=24)
    assert list(preds.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(preds) == 24
    expected = [start + t * step for t in range(16, 40)]
    assert preds["timestamp"].tolist() == expected
    assert preds["timestamp"].is_unique
    assert (preds["active_vm_count"] == 5).all()
    assert preds["aggregate_cpu_utilization"].between(0, 100).all()
    assert np.isfinite(preds["aggregate_cpu_utilization"]).all()

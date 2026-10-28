# Cloud Workload Forecasting

Forecasting the next 2 hours of fleet CPU usage for 10,000 Azure VMs, and turning the forecast into a capacity recommendation.

**Result:** the final method (a blend of last value and training mean) beats the last value baseline by about 15% on the holdout (MAE 0.124 vs 0.147, normalized MAE 0.848) and is almost unbiased. A safety margin of 4.7%, sized only from validation errors, covered all 24 holdout points.

The full write up is in [`reports/final_report.md`](reports/final_report.md). The analysis is in [`notebooks/exploration.ipynb`](notebooks/exploration.ipynb). The original assignment is in [`ASSIGNMENT.md`](ASSIGNMENT.md).

## Install

Python 3.11 or newer.

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Data

Download `data-v1.tar.gz` from the [data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1) and extract it into `challenge_data/`:

```bash
mkdir -p challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

The data files are not committed. Only `challenge_data/vm_cpu_readings.csv.gz` is needed to produce the predictions.

## Generate the predictions

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

This builds both targets from the raw readings, keeps the last 24 timestamps as the holdout, trains on everything before them and writes `predictions.csv` (24 rows: `timestamp,active_vm_count,aggregate_cpu_utilization`). Nothing is hard coded to this dataset: the holdout is always the last `--horizon` timestamps (default 24), so the same command works on a smaller dataset with the same columns.

## Reproduce the evaluation

```bash
python -m cwfc.evaluate --data challenge_data
```

Prints the validation table for every method, the holdout scores and the safety margin table. To rebuild the report figures:

```bash
python reports/make_figures.py --data challenge_data
```

## Tests

```bash
python -m pytest -q
```

Covers target construction (each VM counted once, missing readings ignored), the chronological split, the baselines and blend, that lag features only use the past, that validation never reaches the holdout, and an end to end run on a tiny synthetic dataset.

## Main decisions

**Targets.** `active_vm_count` is the number of distinct VMs with a valid reading per timestamp. `aggregate_cpu_utilization` is the mean of the VM level `avg_cpu` per timestamp, clipped to [0, 100]. The data had no missing values, no duplicate (vm, timestamp) pairs, no gaps (all 133 steps are exactly 300 s) and all values within [0, 100], so no cleaning or resampling was needed. If readings were missing, they would be ignored rather than filled in.

**Active VM count.** It is exactly 10,000 at every timestamp, because the challenge only kept VMs with all 134 readings. The last value forecast is therefore exact. This says nothing about real churn, see limitations.

**Validation.** Rolling origin with an expanding window, only inside the first 110 points: 4 rounds starting at points 50, 62, 74 and 86, each forecasting the next 24 points (the same 2 hour horizon as the real task). The holdout was only opened once, after the final method was chosen. The starter `rolling_origin_splits` was not used as is, because its defaults (1 week of training, 1 day horizon) produce zero rounds on 134 points. Seasonal naive was not used, because 11 hours cannot contain a daily cycle.

**Methods compared** (mean over the 4 validation rounds):

| method | MAE | RMSE | bias |
|---|---|---|---|
| **blend (final)** | **0.198** | **0.228** | +0.036 |
| last value | 0.206 | 0.239 | +0.067 |
| train mean | 0.214 | 0.243 | +0.005 |
| ridge on lag features | 0.247 | 0.278 | +0.081 |
| moving average 1 h | 0.254 | 0.282 | +0.077 |

The blend averages the last value and the training mean: it starts from the current level but expects a pull back toward the normal level, which matches what the exploration showed. Ridge regression on lag features (5 to 60 minutes back plus a 1 hour rolling mean) did not beat the simple rules: the training windows are tiny (38 to 74 rows) and the series moves inside about 0.8 percentage points with no repeating pattern.

**Holdout** (scored once, choice fixed in advance):

| method | MAE | RMSE | bias |
|---|---|---|---|
| ridge on lag features | 0.106 | 0.142 | -0.035 |
| moving average 1 h | 0.123 | 0.161 | -0.053 |
| **blend (final)** | **0.124** | **0.154** | **+0.020** |
| last value | 0.147 | 0.185 | -0.104 |
| train mean | 0.175 | 0.210 | +0.145 |

Ridge scored best on this single window but lost on every validation round, so the choice was not changed after seeing the holdout.

**Recommendation.** Plan fleet capacity at forecast + 4.7% (the 95th percentile of the blend's validation misses). Use the 90% margin (+3.5%) for delay insensitive workloads, which can queue. Size individual VMs on their peaks, not on the fleet average: inside 5 minutes max CPU is about 2x the average, and lifetime p95 peaks are 5 to 7x the average.

## Limitations

1. Only 11 hours of data: no daily or weekly cycles can be learned.
2. The selection filter keeps only VMs alive for the whole window, so lifetimes look longer and the fleet more stable than in reality.
3. The target is a mean of percentages, so a 2 core VM counts as much as a 24 core VM. It measures how full a typical VM is, not total CPU demand.
4. The safety margin comes from 96 validation misses in one window and should be recalculated as more data arrives.

## Repository layout

```text
src/cwfc/
  data.py        load readings, build targets, chronological split
  forecast.py    baselines, blend, ridge on lag features
  predict.py     command that writes predictions.csv
  evaluate.py    validation, holdout scores, safety margin
  metrics.py     starter metrics (mae, rmse, smape, normalized mae)
  baselines.py   starter baselines
  validation.py  starter rolling origin helper
notebooks/exploration.ipynb   data understanding, workload analysis, modeling
reports/final_report.md       final report
reports/make_figures.py       rebuilds reports/figures
tests/                        unit and end to end tests
predictions.csv               submitted predictions
```

#!/usr/bin/env python3
"""
APLAND fixed-12-day CNN-BiLSTM-Attention training pipeline.

Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

Purpose
-------
This script corrects the temporal-cadence issue in the submitted APLAND
pipeline. The raw EGMS file contains mixed 6-, 12-, and 24-day intervals.
The script constructs an explicit 12-day grid anchored at the first valid
acquisition, keeps only acquisitions exactly on that grid, inserts absent
grid epochs as NaN, reports missingness before imputation, and then applies
causal forward filling to model inputs only. A target transition is eligible
only when both its target epoch and immediately preceding epoch were genuinely
observed. Thus, neither a synthetic zero increment at an imputed epoch nor a
24-day accumulated increment after a gap is used as a 12-day learning target.

It also corrects the recent-time sampling equation. For p > 1,
    v = 1 - (1-u)**p
is biased toward 1 (recent targets). The previously used u**p is biased
toward 0 (older targets).

Outputs
-------
cadence_raw_intervals.csv
cadence_summary.json
fixed12_grid.csv
training_history.csv
training_history.png
test_predictions_fixed12.csv
test_metrics_fixed12.json
best_fixed12.keras
table9_sampling_robustness.csv      (optional; --revision-table-evaluation)
table10_active_subset.csv           (optional; --revision-table-evaluation)
table13_intra_corridor.csv          (optional; --revision-table-evaluation)
revision_table_evaluation_audit.json (optional; --revision-table-evaluation)

Example
-------
python APLAND_fixed12_training_revision.py \
  --csv "data/EGMS_L2b.csv" \
  --evaluate-existing-model "outputs/fixed12_training/best_fixed12.keras" \
  --revision-table-evaluation \
  --output-dir "outputs/fixed12_revision_tables"

Use --cadence-only to produce the temporal audit without training.
Use --evaluate-existing-model PATH --revision-table-evaluation to evaluate
an already-trained checkpoint for Tables 9, 10, and 13 without retraining it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class Config:
    lookback: int = 24
    horizon_days: int = 12
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    batch_size: int = 1024
    epochs: int = 60
    steps_per_epoch: int = 600
    validation_steps: int = 120
    test_samples: int = 200_000
    bias_power: float = 2.0
    winsor_x: float = 6.0
    label_clip_sigma: float = 3.0
    huber_delta: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit EGMS cadence and train on an explicit fixed 12-day grid."
    )
    p.add_argument("--csv", required=True, help="Path to the original EGMS CSV.")
    p.add_argument("--output-dir", default="fixed12_outputs")
    p.add_argument(
        "--anchor",
        default=None,
        help="Grid anchor in YYYYMMDD form. Default: earliest acquisition.",
    )
    p.add_argument("--cadence-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-samples", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--validation-steps", type=int, default=120)
    p.add_argument("--bias-power", type=float, default=2.0)
    p.add_argument(
        "--revision-table-evaluation", "--table9-sampling-robustness",
        dest="revision_table_evaluation",
        action="store_true",
        help=("Evaluate the same fitted model for Tables 9, 10, and 13: sampling "
              "robustness, the active 10%% subset, and intra-corridor regions."),
    )
    p.add_argument(
        "--evaluate-existing-model",
        default=None,
        metavar="MODEL.keras",
        help=("Evaluate a saved model for Tables 9, 10, and 13 without training. "
              "Requires --revision-table-evaluation."),
    )
    p.add_argument("--robustness-samples", type=int, default=200_000)
    p.add_argument("--full-coverage-samples", type=int, default=1_000_000)
    p.add_argument("--active-fraction", type=float, default=0.10)
    p.add_argument("--regional-samples", type=int, default=100_000)
    p.add_argument("--eastern-fraction", type=float, default=0.20)
    p.add_argument(
        "--sampling-seed", type=int, default=None,
        help=("Fixed sample-generation seed for Table 9 and 13. By default, "
              "uses --seed so the recency-biased sample matches the headline test sample."),
    )
    return p.parse_args()


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def is_date_column(name: object) -> bool:
    return bool(DATE_RE.fullmatch(str(name)))


def find_date_columns(columns: pd.Index) -> list[str]:
    date_cols = [str(c) for c in columns if is_date_column(c)]
    if not date_cols:
        raise ValueError("No date columns in YYYYMMDD format were found.")
    parsed = pd.to_datetime(date_cols, format="%Y%m%d", errors="raise")
    order = np.argsort(parsed.values)
    sorted_cols = [date_cols[i] for i in order]
    if len(set(sorted_cols)) != len(sorted_cols):
        raise ValueError("Duplicate acquisition-date columns were detected.")
    return sorted_cols


def audit_raw_cadence(date_cols: list[str], out_dir: Path) -> dict:
    dates = pd.DatetimeIndex(pd.to_datetime(date_cols, format="%Y%m%d"))
    delta_days = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    intervals = pd.DataFrame(
        {
            "start_date": dates[:-1].strftime("%Y-%m-%d"),
            "end_date": dates[1:].strftime("%Y-%m-%d"),
            "delta_days": delta_days,
        }
    )
    intervals.to_csv(out_dir / "cadence_raw_intervals.csv", index=False)
    counts = Counter(int(x) for x in delta_days)
    summary = {
        "raw_epoch_count": int(len(dates)),
        "raw_interval_count": int(len(delta_days)),
        "first_raw_date": dates[0].strftime("%Y-%m-%d"),
        "last_raw_date": dates[-1].strftime("%Y-%m-%d"),
        "raw_span_days": int((dates[-1] - dates[0]).days),
        "interval_frequency": {str(k): int(v) for k, v in sorted(counts.items())},
        "minimum_interval_days": int(delta_days.min()),
        "maximum_interval_days": int(delta_days.max()),
        "mean_interval_days": float(delta_days.mean()),
        "median_interval_days": float(np.median(delta_days)),
        "is_strictly_12_day": bool(np.all(delta_days == 12)),
    }
    return summary


def construct_fixed_grid(
    date_cols: list[str], anchor_text: str | None, horizon_days: int
) -> tuple[pd.DatetimeIndex, list[str], list[str]]:
    raw_dates = pd.DatetimeIndex(pd.to_datetime(date_cols, format="%Y%m%d"))
    anchor = (
        pd.to_datetime(anchor_text, format="%Y%m%d")
        if anchor_text
        else raw_dates.min()
    )
    if anchor not in raw_dates:
        raise ValueError(f"The requested anchor {anchor.date()} is not a raw acquisition.")
    if anchor > raw_dates.max():
        raise ValueError("Grid anchor occurs after the final raw acquisition.")

    n_steps = int((raw_dates.max() - anchor).days // horizon_days)
    grid = pd.DatetimeIndex(
        [anchor + pd.Timedelta(days=horizon_days * i) for i in range(n_steps + 1)]
    )
    raw_set = set(raw_dates)
    present = [d.strftime("%Y%m%d") for d in grid if d in raw_set]
    absent = [d.strftime("%Y%m%d") for d in grid if d not in raw_set]
    return grid, present, absent


def load_and_regrid(
    csv_path: Path, out_dir: Path, cfg: Config, anchor_text: str | None
) -> tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex, np.ndarray, dict]:
    print(f"[INFO] Reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    for required in ("latitude", "longitude"):
        if required not in df.columns:
            raise ValueError(f"Required coordinate column '{required}' is absent.")

    date_cols = find_date_columns(df.columns)
    summary = audit_raw_cadence(date_cols, out_dir)
    grid, present, absent = construct_fixed_grid(
        date_cols, anchor_text, cfg.horizon_days
    )

    raw_dates = pd.DatetimeIndex(pd.to_datetime(date_cols, format="%Y%m%d"))
    grid_set = set(grid)
    excluded_off_grid = [d.strftime("%Y%m%d") for d in raw_dates if d not in grid_set]

    # Rename raw date headers to Timestamp, then reindex to the complete grid.
    temporal = df[date_cols].copy()
    temporal.columns = raw_dates
    fixed = temporal.reindex(columns=grid)

    missing_cells_before = int(fixed.isna().to_numpy().sum())
    total_expected_cells = int(fixed.shape[0] * fixed.shape[1])
    missing_rate_before = (
        missing_cells_before / total_expected_cells if total_expected_cells else math.nan
    )
    all_missing_epochs = [
        d.strftime("%Y%m%d") for d in grid if fixed[d].isna().all()
    ]

    grid_table = pd.DataFrame(
        {
            "grid_index": np.arange(len(grid)),
            "expected_date": grid.strftime("%Y-%m-%d"),
            "raw_column_present": [d in set(raw_dates) for d in grid],
            "all_values_missing_before_imputation": [fixed[d].isna().all() for d in grid],
        }
    )
    grid_table.to_csv(out_dir / "fixed12_grid.csv", index=False)

    observed_epoch_mask = np.asarray([d in set(raw_dates) for d in grid], dtype=bool)
    eligible_transition_mask = observed_epoch_mask.copy()
    eligible_transition_mask[0] = False
    eligible_transition_mask[1:] &= observed_epoch_mask[:-1]
    excluded_target_dates = grid[~eligible_transition_mask & (np.arange(len(grid)) > 0)]

    summary.update(
        {
            "grid_anchor": grid[0].strftime("%Y-%m-%d"),
            "grid_last_date": grid[-1].strftime("%Y-%m-%d"),
            "grid_horizon_days": cfg.horizon_days,
            "fixed_grid_epoch_count": int(len(grid)),
            "raw_columns_retained_on_grid": int(len(present)),
            "raw_columns_excluded_off_grid": int(len(excluded_off_grid)),
            "excluded_off_grid_dates": excluded_off_grid,
            "expected_grid_epochs_absent_from_raw": absent,
            "all_missing_grid_epochs_before_imputation": all_missing_epochs,
            "missing_cells_before_imputation": missing_cells_before,
            "expected_cells_on_fixed_grid": total_expected_cells,
            "missing_rate_before_imputation": float(missing_rate_before),
            "target_rule": (
                "A target transition is used only if both t and t-1 are raw observed "
                "acquisitions; forward-filled values are input-only."
            ),
            "target_dates_excluded_due_to_missing_endpoint": [
                d.strftime("%Y%m%d") for d in excluded_target_dates
            ],
        }
    )
    with (out_dir / "cadence_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("[CADENCE]", json.dumps(summary, indent=2, ensure_ascii=False))

    # Causal imputation: only past information may fill a missing grid epoch.
    fixed_ff = fixed.ffill(axis=1)
    leading_missing = int(fixed_ff.isna().to_numpy().sum())
    if leading_missing:
        print(
            f"[WARN] {leading_missing:,} leading cells remain after forward fill; "
            "they are replaced by the fixed reference value 0."
        )
    fixed_clean = fixed_ff.fillna(0.0).astype(np.float32)
    return df, fixed_clean.to_numpy(np.float32), grid, observed_epoch_mask, summary


def corrected_recent_times(
    rng: np.random.Generator,
    t_low: int,
    t_high: int,
    size: int,
    bias_power: float,
) -> np.ndarray:
    """Draw integer indices in [t_low, t_high), biased toward recent times."""
    if not t_low < t_high:
        raise ValueError(f"Invalid target range [{t_low}, {t_high}).")
    if bias_power <= 0:
        raise ValueError("bias_power must be positive.")
    u = rng.random(size)
    if bias_power == 1.0:
        v = u
    else:
        v = 1.0 - np.power(1.0 - u, bias_power)
    idx = t_low + np.floor((t_high - t_low) * v).astype(np.int64)
    return np.minimum(idx, t_high - 1)


def sample_eligible_times(
    rng: np.random.Generator,
    eligible_times: np.ndarray,
    size: int,
    bias_power: float,
) -> np.ndarray:
    """Sample from sorted valid target indices with corrected recent bias."""
    eligible_times = np.asarray(eligible_times, dtype=np.int64)
    if eligible_times.ndim != 1 or eligible_times.size == 0:
        raise ValueError("No eligible target epochs are available in this split.")
    ranks = corrected_recent_times(rng, 0, eligible_times.size, size, bias_power)
    return eligible_times[ranks]


def detrended_channels(window: np.ndarray, cfg: Config) -> np.ndarray:
    t = np.arange(cfg.lookback, dtype=np.float32)
    A = np.column_stack((t, np.ones_like(t)))
    slope, intercept = np.linalg.lstsq(A, window, rcond=None)[0]
    detrended = window - (slope * t + intercept)
    first_difference = np.diff(detrended, prepend=detrended[0])
    channels = np.stack((detrended, first_difference), axis=-1)
    return np.clip(channels, -cfg.winsor_x, cfg.winsor_x).astype(np.float32)


def make_batch(
    arr_norm: np.ndarray,
    eligible_times: np.ndarray,
    cfg: Config,
    rng: np.random.Generator,
    clip_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    times = sample_eligible_times(
        rng, eligible_times, cfg.batch_size, cfg.bias_power
    )
    points = rng.integers(0, arr_norm.shape[0], size=cfg.batch_size)
    X = np.empty((cfg.batch_size, cfg.lookback, 2), dtype=np.float32)
    y = np.empty((cfg.batch_size, 1), dtype=np.float32)
    for i, (point, target_t) in enumerate(zip(points, times)):
        X[i] = detrended_channels(
            arr_norm[point, target_t - cfg.lookback : target_t], cfg
        )
        y[i, 0] = arr_norm[point, target_t] - arr_norm[point, target_t - 1]
    if clip_labels:
        batch_std = float(np.std(y) + 1e-8)
        limit = max(0.5, cfg.label_clip_sigma * batch_std)
        y = np.clip(y, -limit, limit)
    return X, y


def make_tf_dataset(
    arr_norm: np.ndarray,
    eligible_times: np.ndarray,
    cfg: Config,
    seed: int,
    clip_labels: bool,
):
    import tensorflow as tf

    def generator():
        rng = np.random.default_rng(seed)
        while True:
            yield make_batch(arr_norm, eligible_times, cfg, rng, clip_labels)

    signature = (
        tf.TensorSpec((None, cfg.lookback, 2), tf.float32),
        tf.TensorSpec((None, 1), tf.float32),
    )
    return (
        tf.data.Dataset.from_generator(generator, output_signature=signature)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )


def make_fixed_test(
    arr_norm: np.ndarray,
    eligible_times: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    times = sample_eligible_times(
        rng, eligible_times, cfg.test_samples, cfg.bias_power
    )
    points = rng.integers(0, arr_norm.shape[0], size=cfg.test_samples)
    X = np.empty((cfg.test_samples, cfg.lookback, 2), dtype=np.float32)
    y = np.empty((cfg.test_samples, 1), dtype=np.float32)
    for i, (point, target_t) in enumerate(zip(points, times)):
        X[i] = detrended_channels(
            arr_norm[point, target_t - cfg.lookback : target_t], cfg
        )
        y[i, 0] = arr_norm[point, target_t] - arr_norm[point, target_t - 1]
    return X, y, points, times


def build_model(cfg: Config):
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = layers.Input((cfg.lookback, 2), name="fixed12_input")
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inputs)
    x = layers.Conv1D(64, 5, padding="causal", activation="gelu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)
    )(x)
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)
    )(x)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x, x)
    x = layers.Add()([x, attention])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    output = layers.Dense(1, name="normalized_12day_increment")(x)
    return tf.keras.Model(inputs, output, name="CNN_BiLSTM_Attention_fixed12")


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum(np.square(y_true - y_pred), dtype=np.float64))
    total = float(
        np.sum(np.square(y_true - np.mean(y_true)), dtype=np.float64)
    )
    return float(1.0 - residual / total) if total > 0 else math.nan


def evaluate_and_save(
    model,
    X_test: np.ndarray,
    y_norm: np.ndarray,
    points: np.ndarray,
    times: np.ndarray,
    arr_values: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    grid: pd.DatetimeIndex,
    out_dir: Path,
) -> dict:
    pred_norm = model.predict(X_test, batch_size=4096, verbose=1).reshape(-1)
    true_norm = y_norm.reshape(-1)
    point_std = stds[points, 0]
    true_delta = true_norm * point_std
    pred_delta = pred_norm * point_std
    last_level = arr_values[points, times - 1]
    true_level = last_level + true_delta
    pred_level = last_level + pred_delta

    def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        error = y_pred - y_true
        return {
            "mae_mm": float(np.mean(np.abs(error))),
            "rmse_mm": float(np.sqrt(np.mean(np.square(error)))),
            "r2": r2_score_np(y_true, y_pred),
        }

    result = {
        "increment_space": metrics(true_delta, pred_delta),
        "reconstructed_level_space": metrics(true_level, pred_level),
        "test_sample_count": int(len(points)),
        "sampling": "corrected recent-biased draw: 1-(1-u)^p",
    }
    with (out_dir / "test_metrics_fixed12.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    prediction_table = pd.DataFrame(
        {
            "point_index": points,
            "target_epoch_index": times,
            "target_date": grid[times].strftime("%Y-%m-%d"),
            "last_observed_date": grid[times - 1].strftime("%Y-%m-%d"),
            "horizon_days": (grid[times] - grid[times - 1]).days,
            "true_increment_mm": true_delta,
            "predicted_increment_mm": pred_delta,
            "last_observed_level_mm": last_level,
            "true_level_mm": true_level,
            "predicted_level_mm": pred_level,
        }
    )
    prediction_table.to_csv(out_dir / "test_predictions_fixed12.csv", index=False)
    return result


def sample_test_indices(
    eligible_times: np.ndarray,
    n_points: int,
    n_samples: int,
    bias_power: float,
    sampling_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a reproducible (point, target-time) sample for test evaluation.

    The draw order matches ``make_fixed_test``: draw all temporal indices first,
    then draw spatial point indices.  This preserves the headline p=2, n=200k
    sample when sampling_seed is 42.
    """
    if n_samples <= 0:
        raise ValueError("The requested test sample size must be positive.")
    rng = np.random.default_rng(sampling_seed)
    times = sample_eligible_times(rng, eligible_times, n_samples, bias_power)
    points = rng.integers(0, n_points, size=n_samples, dtype=np.int64)
    return points, times


def evaluate_sample_in_batches(
    model,
    arr_norm: np.ndarray,
    arr_values: np.ndarray,
    stds: np.ndarray,
    points: np.ndarray,
    times: np.ndarray,
    cfg: Config,
    batch_size: int = 4096,
    return_predictions: bool = False,
) -> dict:
    """Compute pooled metrics without allocating all 1M input windows at once."""
    true_delta_parts, pred_delta_parts = [], []
    true_level_parts, pred_level_parts = [], []

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        p_batch = points[start:stop]
        t_batch = times[start:stop]
        X = np.empty((len(p_batch), cfg.lookback, 2), dtype=np.float32)
        y_norm = np.empty(len(p_batch), dtype=np.float32)
        for i, (point, target_t) in enumerate(zip(p_batch, t_batch)):
            X[i] = detrended_channels(
                arr_norm[point, target_t - cfg.lookback : target_t], cfg
            )
            y_norm[i] = arr_norm[point, target_t] - arr_norm[point, target_t - 1]

        pred_norm = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
        point_std = stds[p_batch, 0]
        true_delta = y_norm * point_std
        pred_delta = pred_norm * point_std
        last_level = arr_values[p_batch, t_batch - 1]
        true_level = last_level + true_delta
        pred_level = last_level + pred_delta

        true_delta_parts.append(true_delta.astype(np.float64, copy=False))
        pred_delta_parts.append(pred_delta.astype(np.float64, copy=False))
        true_level_parts.append(true_level.astype(np.float64, copy=False))
        pred_level_parts.append(pred_level.astype(np.float64, copy=False))

    true_delta = np.concatenate(true_delta_parts)
    pred_delta = np.concatenate(pred_delta_parts)
    true_level = np.concatenate(true_level_parts)
    pred_level = np.concatenate(pred_level_parts)

    def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, float]:
        error = y_pred - y_true
        sst = float(np.sum(np.square(y_true - np.mean(y_true)), dtype=np.float64))
        return ({
            "mae_mm": float(np.mean(np.abs(error))),
            "rmse_mm": float(np.sqrt(np.mean(np.square(error)))),
            "r2": float(1.0 - np.sum(np.square(error), dtype=np.float64) / sst)
                  if sst > 0 else math.nan,
        }, sst)

    delta_metrics, delta_sst = pooled_metrics(true_delta, pred_delta)
    level_metrics, level_sst = pooled_metrics(true_level, pred_level)
    result = {
        "increment_space": delta_metrics,
        "reconstructed_level_space": level_metrics,
        "target_sst_delta_mm2": delta_sst,
        "target_variance_delta_mm2": delta_sst / len(points),
        "target_sst_level_mm2": level_sst,
        "target_variance_level_mm2": level_sst / len(points),
        "test_sample_count": int(len(points)),
    }
    if return_predictions:
        result["prediction_arrays"] = {
            "true_delta": true_delta,
            "pred_delta": pred_delta,
            "true_level": true_level,
            "pred_level": pred_level,
        }
    return result


def evaluate_revision_tables(
    model,
    df: pd.DataFrame,
    arr_values: np.ndarray,
    observed_epoch_mask: np.ndarray,
    cfg: Config,
    out_dir: Path,
    sampling_seed: int = 42,
    robustness_samples: int = 200_000,
    full_coverage_samples: int = 1_000_000,
    active_fraction: float = 0.10,
    regional_samples: int = 100_000,
    eastern_fraction: float = 0.20,
) -> pd.DataFrame:
    """Evaluate the current revision's Tables 9, 10, and 13 without retraining.

    Table 9 changes only the test-window sampling regime. Table 10 selects the
    top active fraction from the same fixed recency-biased headline sample.
    Table 13 evaluates 100k windows in each region using one shared set of
    test epochs and the existing all-corridor model; it is not spatial hold-out.
    """
    if robustness_samples <= 0 or full_coverage_samples <= 0 or regional_samples <= 0:
        raise ValueError("All requested test sample counts must be positive.")
    if not 0.0 < active_fraction <= 1.0:
        raise ValueError("active_fraction must be in (0, 1].")
    if not 0.0 < eastern_fraction < 1.0:
        raise ValueError("eastern_fraction must be in (0, 1).")
    if "longitude" not in df.columns:
        raise ValueError("Table 13 requires a 'longitude' column in the source CSV.")
    if len(df) != arr_values.shape[0]:
        raise ValueError("Source CSV point count does not match the regridded data.")

    T = arr_values.shape[1]
    train_end = int(T * cfg.train_fraction)
    means = np.mean(
        arr_values[:, :train_end], axis=1, keepdims=True, dtype=np.float64
    ).astype(np.float32)
    stds = np.std(
        arr_values[:, :train_end], axis=1, keepdims=True, dtype=np.float64
    ).astype(np.float32)
    means[~np.isfinite(means)] = 0.0
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    arr_norm = ((arr_values - means) / stds).astype(np.float32)

    transition_observed = observed_epoch_mask.copy()
    transition_observed[0] = False
    transition_observed[1:] &= observed_epoch_mask[:-1]
    validation_end = int(T * (cfg.train_fraction + cfg.validation_fraction))
    test_indices = np.arange(max(validation_end, cfg.lookback), T, dtype=np.int64)
    eligible_test_times = test_indices[transition_observed[test_indices]]
    if eligible_test_times.size == 0:
        raise ValueError("No eligible target epochs are available in the test interval.")

    regimes = [
        ("recency-biased", 2.0, robustness_samples),
        ("uniform", 1.0, robustness_samples),
        ("full-coverage proxy", 1.0, full_coverage_samples),
    ]
    table9_rows = []
    headline_prediction_arrays = None
    for regime, power, n_samples in regimes:
        points, times = sample_test_indices(
            eligible_test_times, arr_values.shape[0], n_samples, power, sampling_seed
        )
        metrics = evaluate_sample_in_batches(
            model, arr_norm, arr_values, stds, points, times, cfg,
            return_predictions=(regime == "recency-biased"),
        )
        if regime == "recency-biased":
            headline_prediction_arrays = metrics.pop("prediction_arrays")
        delta = metrics["increment_space"]
        level = metrics["reconstructed_level_space"]
        table9_rows.append({
            "regime": regime,
            "temporal_sampling_power": power,
            "n_test_windows": n_samples,
            "sample_generation_seed": sampling_seed,
            "training_seed": cfg.seed,
            "MAE_delta_mm": delta["mae_mm"],
            "RMSE_delta_mm": delta["rmse_mm"],
            "R2_delta": delta["r2"],
            "MAE_level_mm": level["mae_mm"],
            "RMSE_level_mm": level["rmse_mm"],
            "R2_level": level["r2"],
            "target_variance_delta_mm2": metrics["target_variance_delta_mm2"],
            "target_SST_delta_mm2": metrics["target_sst_delta_mm2"],
        })

    table9 = pd.DataFrame(table9_rows)
    table9.to_csv(out_dir / "table9_sampling_robustness.csv", index=False)

    # Table 10: top active fraction by |observed increment| from the same
    # 200k p=2 headline sample. Stable sorting makes tie handling reproducible.
    if headline_prediction_arrays is None:
        raise RuntimeError("Headline predictions were not retained for Table 10.")
    active_data = headline_prediction_arrays
    n_active = max(1, int(round(len(active_data["true_delta"]) * active_fraction)))
    active_order = np.argsort(-np.abs(active_data["true_delta"]), kind="stable")
    active_idx = active_order[:n_active]

    def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        error = y_pred - y_true
        sst = float(np.sum(np.square(y_true - np.mean(y_true)), dtype=np.float64))
        return {
            "MAE_mm": float(np.mean(np.abs(error))),
            "RMSE_mm": float(np.sqrt(np.mean(np.square(error)))),
            "R2": float(1.0 - np.sum(np.square(error), dtype=np.float64) / sst)
                  if sst > 0 else math.nan,
        }

    table10_rows = []
    for subset, idx in (("Overall test", np.arange(len(active_data["true_delta"]))),
                        ("Active subset", active_idx)):
        d = summarize(active_data["true_delta"][idx], active_data["pred_delta"][idx])
        level = summarize(active_data["true_level"][idx], active_data["pred_level"][idx])
        table10_rows.append({
            "subset": subset,
            "definition": ("All fixed headline test windows" if subset == "Overall test"
                           else f"Top {100 * active_fraction:g}% by absolute observed increment"),
            "n_test_windows": int(len(idx)),
            "MAE_delta_mm": d["MAE_mm"],
            "RMSE_delta_mm": d["RMSE_mm"],
            "R2_delta": d["R2"],
            "MAE_level_mm": level["MAE_mm"],
            "RMSE_level_mm": level["RMSE_mm"],
            "R2_level": level["R2"],
        })
    table10 = pd.DataFrame(table10_rows)
    table10.to_csv(out_dir / "table10_active_subset.csv", index=False)

    # Table 13: create an exact 80/20 point partition by longitude rank. The
    # same temporal test indices are used in both regions; points are sampled
    # uniformly within each region. Both groups remain in training for this
    # intra-corridor heterogeneity analysis.
    lons = pd.to_numeric(df["longitude"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(lons).all():
        raise ValueError("Table 13 longitude values contain missing/non-finite entries.")
    n_points = len(lons)
    n_east = int(round(n_points * eastern_fraction))
    if n_east <= 0 or n_east >= n_points:
        raise ValueError("The requested eastern fraction yields an empty region.")
    order = np.argsort(lons, kind="stable")
    interior_points = order[:-n_east]
    eastern_points = order[-n_east:]

    time_rng = np.random.default_rng(sampling_seed)
    shared_times = sample_eligible_times(
        time_rng, eligible_test_times, regional_samples, bias_power=2.0
    )
    table13_rows = []
    regional_results = {}
    for region, region_points, seed_offset in (
        ("Interior region", interior_points, 1),
        ("Eastern block", eastern_points, 2),
    ):
        point_rng = np.random.default_rng(sampling_seed + seed_offset)
        sampled_points = region_points[
            point_rng.integers(0, len(region_points), size=regional_samples)
        ]
        metrics = evaluate_sample_in_batches(
            model, arr_norm, arr_values, stds, sampled_points, shared_times, cfg
        )
        delta = metrics["increment_space"]
        level = metrics["reconstructed_level_space"]
        row = {
            "region": region,
            "region_fraction": (1.0 - eastern_fraction if region == "Interior region"
                                else eastern_fraction),
            "n_points_in_region": int(len(region_points)),
            "n_test_windows": int(regional_samples),
            "temporal_sampling_power": 2.0,
            "sample_generation_seed": sampling_seed,
            "training_seed": cfg.seed,
            "MAE_delta_mm": delta["mae_mm"],
            "RMSE_delta_mm": delta["rmse_mm"],
            "R2_delta": delta["r2"],
            "MAE_level_mm": level["mae_mm"],
            "RMSE_level_mm": level["rmse_mm"],
            "R2_level": level["r2"],
        }
        table13_rows.append(row)
        regional_results[region] = row

    interior = regional_results["Interior region"]
    east = regional_results["Eastern block"]
    table13_rows.append({
        "region": "80/20 area-weighted summary",
        "region_fraction": 1.0,
        "n_points_in_region": n_points,
        "n_test_windows": None,
        "temporal_sampling_power": None,
        "sample_generation_seed": sampling_seed,
        "training_seed": cfg.seed,
        "MAE_delta_mm": ((1.0 - eastern_fraction) * interior["MAE_delta_mm"]
                         + eastern_fraction * east["MAE_delta_mm"]),
        "RMSE_delta_mm": math.sqrt(
            (1.0 - eastern_fraction) * interior["RMSE_delta_mm"] ** 2
            + eastern_fraction * east["RMSE_delta_mm"] ** 2
        ),
        "R2_delta": None,
        "MAE_level_mm": ((1.0 - eastern_fraction) * interior["MAE_level_mm"]
                         + eastern_fraction * east["MAE_level_mm"]),
        "RMSE_level_mm": math.sqrt(
            (1.0 - eastern_fraction) * interior["RMSE_level_mm"] ** 2
            + eastern_fraction * east["RMSE_level_mm"] ** 2
        ),
        "R2_level": None,
    })
    table13 = pd.DataFrame(table13_rows)
    table13.to_csv(out_dir / "table13_intra_corridor.csv", index=False)

    audit = {
        "purpose": "Evaluation code for revised manuscript Tables 9, 10, and 13",
        "model_weights_changed": False,
        "training_repeated": False,
        "training_seed": cfg.seed,
        "sample_generation_seed": sampling_seed,
        "test_eligible_target_epochs": int(eligible_test_times.size),
        "table9": {
            "metric_aggregation": "pooled globally over paired windows per regime",
            "r2_delta_definition": "1 - SSE_delta / SST_delta within each regime",
            "rows": table9_rows,
        },
        "table10": {
            "active_rule": f"top {active_fraction:.6g} fraction ranked by absolute observed increment",
            "uses_same_headline_test_sample": True,
            "rows": table10_rows,
        },
        "table13": {
            "analysis_type": "intra-corridor regional heterogeneity; not spatial hold-out",
            "region_rule": "easternmost points by stable longitude rank",
            "eastern_fraction": eastern_fraction,
            "shared_test_epochs_between_regions": True,
            "region_rows": table13_rows[:2],
            "area_weighted_summary": table13_rows[2],
        },
    }
    (out_dir / "revision_table_evaluation_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print("[TABLE 9] Sampling robustness")
    print(table9.to_string(index=False))
    print("[TABLE 10] Active subset")
    print(table10.to_string(index=False))
    print("[TABLE 13] Intra-corridor regional evaluation")
    print(table13.to_string(index=False))
    return table9


def train_pipeline(
    df: pd.DataFrame,
    arr_values: np.ndarray,
    grid: pd.DatetimeIndex,
    observed_epoch_mask: np.ndarray,
    cfg: Config,
    out_dir: Path,
    revision_table_evaluation: bool = False,
    sampling_seed: int = 42,
    robustness_samples: int = 200_000,
    full_coverage_samples: int = 1_000_000,
    active_fraction: float = 0.10,
    regional_samples: int = 100_000,
    eastern_fraction: float = 0.20,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    tf.random.set_seed(cfg.seed)
    T = arr_values.shape[1]
    train_end = int(T * cfg.train_fraction)
    validation_end = int(T * (cfg.train_fraction + cfg.validation_fraction))
    if train_end <= cfg.lookback:
        raise ValueError("The fixed grid is too short for the chosen lookback.")
    if not train_end < validation_end < T:
        raise ValueError("Invalid chronological split boundaries.")

    # Train-only point-wise statistics after causal grid imputation.
    train_values = arr_values[:, :train_end]
    means = np.mean(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    stds = np.std(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    means[~np.isfinite(means)] = 0.0
    arr_norm = ((arr_values - means) / stds).astype(np.float32)

    transition_observed = observed_epoch_mask.copy()
    transition_observed[0] = False
    transition_observed[1:] &= observed_epoch_mask[:-1]

    def eligible_in(start: int, stop: int) -> np.ndarray:
        indices = np.arange(max(start, cfg.lookback), stop, dtype=np.int64)
        return indices[transition_observed[indices]]

    train_times = eligible_in(cfg.lookback, train_end)
    validation_times = eligible_in(train_end, validation_end)
    test_times = eligible_in(validation_end, T)
    split_audit = {
        "train_eligible_target_count": int(train_times.size),
        "validation_eligible_target_count": int(validation_times.size),
        "test_eligible_target_count": int(test_times.size),
        "train_target_dates": [grid[i].strftime("%Y-%m-%d") for i in train_times],
        "validation_target_dates": [grid[i].strftime("%Y-%m-%d") for i in validation_times],
        "test_target_dates": [grid[i].strftime("%Y-%m-%d") for i in test_times],
    }
    with (out_dir / "eligible_target_audit.json").open("w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)
    print("[TARGET AUDIT]", json.dumps(split_audit, indent=2))

    train_ds = make_tf_dataset(arr_norm, train_times, cfg, cfg.seed, clip_labels=True)
    val_ds = make_tf_dataset(
        arr_norm, validation_times, cfg, cfg.seed + 10_000, clip_labels=True
    )

    model = build_model(cfg)
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=cfg.learning_rate,
        first_decay_steps=max(1, cfg.steps_per_epoch * 5),
    )
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=schedule, weight_decay=cfg.weight_decay, clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=cfg.huber_delta),
        metrics=["mae"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            out_dir / "best_fixed12.keras",
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.CSVLogger(out_dir / "training_history.csv"),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history["loss"], label="training Huber loss")
    plt.plot(history.history["val_loss"], label="validation Huber loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_history.png", dpi=200)
    plt.close()

    X_test, y_test, points, times = make_fixed_test(arr_norm, test_times, cfg)
    results = evaluate_and_save(
        model,
        X_test,
        y_test,
        points,
        times,
        arr_values,
        means,
        stds,
        grid,
        out_dir,
    )
    print("[TEST RESULTS]", json.dumps(results, indent=2))
    if revision_table_evaluation:
        evaluate_revision_tables(
            model=model,
            df=df,
            arr_values=arr_values,
            observed_epoch_mask=observed_epoch_mask,
            cfg=cfg,
            out_dir=out_dir,
            sampling_seed=sampling_seed,
            robustness_samples=robustness_samples,
            full_coverage_samples=full_coverage_samples,
            active_fraction=active_fraction,
            regional_samples=regional_samples,
            eastern_fraction=eastern_fraction,
        )


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
        bias_power=args.bias_power,
    )
    set_seeds(cfg.seed)
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    df, arr_values, grid, observed_epoch_mask, _ = load_and_regrid(
        csv_path, out_dir, cfg, args.anchor
    )
    if args.cadence_only:
        print("[DONE] Cadence audit and fixed-grid construction completed.")
        return
    if args.evaluate_existing_model:
        if not args.revision_table_evaluation:
            raise ValueError(
                "--evaluate-existing-model requires --revision-table-evaluation."
            )
        model_path = Path(args.evaluate_existing_model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path, compile=False)
        evaluate_revision_tables(
            model=model,
            df=df,
            arr_values=arr_values,
            observed_epoch_mask=observed_epoch_mask,
            cfg=cfg,
            out_dir=out_dir,
            sampling_seed=(cfg.seed if args.sampling_seed is None else args.sampling_seed),
            robustness_samples=args.robustness_samples,
            full_coverage_samples=args.full_coverage_samples,
            active_fraction=args.active_fraction,
            regional_samples=args.regional_samples,
            eastern_fraction=args.eastern_fraction,
        )
        print(f"[DONE] Table 9/10/13 evaluation outputs written to {out_dir}")
        return

    train_pipeline(
        df,
        arr_values,
        grid,
        observed_epoch_mask,
        cfg,
        out_dir,
        revision_table_evaluation=args.revision_table_evaluation,
        sampling_seed=(cfg.seed if args.sampling_seed is None else args.sampling_seed),
        robustness_samples=args.robustness_samples,
        full_coverage_samples=args.full_coverage_samples,
        active_fraction=args.active_fraction,
        regional_samples=args.regional_samples,
        eastern_fraction=args.eastern_fraction,
    )
    print(f"[DONE] All outputs written to {out_dir}")


if __name__ == "__main__":
    main()

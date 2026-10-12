#!/usr/bin/env python3
"""MC Dropout uncertainty quantification for the fixed-12-day APLAND model.

The 90% interval is calibrated without test-label leakage: the 90th
percentile of the absolute MC-mean residuals is estimated on a fixed
validation sample and then used as a symmetric half-width on the unchanged
200,000-window test sample saved by the training script.

Example
-------
python APLAND_fixed12_mc_dropout.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation-calibrated MC Dropout evaluation."
    )
    p.add_argument("--csv", type=Path, default=Path("data/EGMS_L2b.csv"))
    p.add_argument(
        "--base-script", type=Path, default=Path("APLAND_fixed12_training.py"),
        help="Public fixed-12-day training script used for preprocessing.",
    )
    p.add_argument(
        "--model", type=Path,
        default=Path("outputs/fixed12_training/best_fixed12.keras"),
    )
    p.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/fixed12_training/test_predictions_fixed12.csv"),
        help="Saved fixed test sample from the training run.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/fixed12_mc_dropout"),
    )
    p.add_argument("--anchor", default=None)
    p.add_argument("--mc-samples", type=int, default=100)
    p.add_argument("--coverage", type=float, default=0.90)
    p.add_argument("--calibration-samples", type=int, default=200_000)
    p.add_argument("--calibration-seed", type=int, default=314159)
    p.add_argument("--batch-size", type=int, default=4096)
    return p.parse_args()


def import_base(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Base training script not found: {path}")
    spec = importlib.util.spec_from_file_location("apland_fixed12_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("CSV", args.csv), ("model", args.model),
        ("predictions", args.predictions), ("base script", args.base_script),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
    if args.mc_samples < 2:
        raise ValueError("--mc-samples must be at least 2.")
    if not 0.0 < args.coverage < 1.0:
        raise ValueError("--coverage must lie strictly between 0 and 1.")
    if args.calibration_samples < 1 or args.batch_size < 1:
        raise ValueError("Sample and batch sizes must be positive.")


def build_windows(arr_norm, point_idx, target_idx, cfg, base):
    X = np.empty((len(point_idx), cfg.lookback, 2), dtype=np.float32)
    for i, (point, target_t) in enumerate(zip(point_idx, target_idx)):
        if target_t < cfg.lookback:
            raise ValueError(f"Target epoch {target_t} has insufficient history.")
        X[i] = base.detrended_channels(
            arr_norm[point, target_t - cfg.lookback:target_t], cfg
        )
    return X


def mc_predict(model, X, n_passes: int, batch_size: int) -> np.ndarray:
    """Run stochastic passes with dropout active and return (passes, samples)."""
    draws = np.empty((n_passes, len(X)), dtype=np.float32)
    for pass_i in range(n_passes):
        chunks = []
        for start in range(0, len(X), batch_size):
            pred = model(X[start:start + batch_size], training=True)
            chunks.append(np.asarray(pred).reshape(-1).astype(np.float32))
        draws[pass_i] = np.concatenate(chunks)
        print(f"  MC pass {pass_i + 1:3d}/{n_passes}", flush=True)
    return draws


def point_metrics(y_level, p_level, y_delta, p_delta):
    return {
        "mae_level_mm": float(mean_absolute_error(y_level, p_level)),
        "rmse_level_mm": float(math.sqrt(mean_squared_error(y_level, p_level))),
        "r2_level": float(r2_score(y_level, p_level)),
        "r2_delta": float(r2_score(y_delta, p_delta)),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = import_base(args.base_script)
    cfg = base.Config(batch_size=args.batch_size)
    base.set_seeds(cfg.seed)

    # Use exactly the preprocessing implementation used for model training.
    _, arr, grid, observed_epochs, cadence = base.load_and_regrid(
        args.csv, args.output_dir, cfg, args.anchor
    )
    n_points, n_epochs = arr.shape
    train_end = int(n_epochs * cfg.train_fraction)
    val_end = int(n_epochs * (cfg.train_fraction + cfg.validation_fraction))

    means = arr[:, :train_end].mean(axis=1, keepdims=True).astype(np.float32)
    stds = arr[:, :train_end].std(axis=1, keepdims=True).astype(np.float32)
    stds[stds == 0.0] = 1.0
    arr_norm = ((arr - means) / stds).astype(np.float32)

    transition_ok = observed_epochs.copy()
    transition_ok[0] = False
    transition_ok[1:] &= observed_epochs[:-1]
    val_times = np.flatnonzero(transition_ok &
                               (np.arange(n_epochs) >= max(train_end, cfg.lookback)) &
                               (np.arange(n_epochs) < val_end))
    if val_times.size == 0:
        raise ValueError("No eligible observed validation transitions were found.")

    # A fixed, reproducible validation-only calibration sample.
    cal_rng = np.random.default_rng(args.calibration_seed)
    cal_t = base.sample_eligible_times(
        cal_rng, val_times, args.calibration_samples, cfg.bias_power
    )
    cal_p = cal_rng.integers(0, n_points, size=args.calibration_samples)
    X_cal = build_windows(arr_norm, cal_p, cal_t, cfg, base)
    y_cal_delta = arr[cal_p, cal_t] - arr[cal_p, cal_t - 1]

    test = pd.read_csv(args.predictions)
    required = {
        "point_index", "target_epoch_index", "horizon_days",
        "true_increment_mm", "true_level_mm",
    }
    missing = required.difference(test.columns)
    if missing:
        raise ValueError(f"Prediction file lacks columns: {sorted(missing)}")
    horizons = sorted(test["horizon_days"].astype(int).unique().tolist())
    if horizons != [cfg.horizon_days]:
        raise ValueError(f"Expected only {cfg.horizon_days}-day targets; got {horizons}")
    if len(test) != cfg.test_samples:
        raise ValueError(
            f"Expected the fixed {cfg.test_samples:,}-window test sample; got {len(test):,}."
        )

    test_p = test["point_index"].to_numpy(np.int64)
    test_t = test["target_epoch_index"].to_numpy(np.int64)
    if np.any(test_p < 0) or np.any(test_p >= n_points):
        raise ValueError("Prediction file contains an invalid point index.")
    if np.any(test_t < val_end) or np.any(test_t >= n_epochs):
        raise ValueError("Prediction file contains an index outside the test interval.")
    if not np.all(transition_ok[test_t]):
        raise ValueError("Prediction file includes an ineligible/imputed target transition.")

    rebuilt_delta = arr[test_p, test_t] - arr[test_p, test_t - 1]
    saved_delta = test["true_increment_mm"].to_numpy(np.float32)
    if not np.allclose(rebuilt_delta, saved_delta, atol=1e-4, rtol=1e-5):
        raise ValueError("Saved test targets do not match the supplied dataset/grid.")
    X_test = build_windows(arr_norm, test_p, test_t, cfg, base)

    import tensorflow as tf
    tf.random.set_seed(cfg.seed)
    model = tf.keras.models.load_model(args.model)

    print("[INFO] Running MC Dropout on validation calibration sample")
    cal_draws_norm = mc_predict(
        model, X_cal, args.mc_samples, args.batch_size
    )
    cal_mean_delta = cal_draws_norm.mean(axis=0) * stds[cal_p, 0]
    cal_abs_residual = np.abs(y_cal_delta - cal_mean_delta)
    half_width = float(np.quantile(cal_abs_residual, args.coverage))
    del cal_draws_norm, X_cal, cal_mean_delta, cal_abs_residual

    print("[INFO] Running MC Dropout on fixed held-out test sample")
    test_draws_norm = mc_predict(
        model, X_test, args.mc_samples, args.batch_size
    )
    test_scale = stds[test_p, 0]
    mean_delta = test_draws_norm.mean(axis=0) * test_scale
    last_level = arr[test_p, test_t - 1]
    mean_level = last_level + mean_delta
    true_delta = saved_delta
    true_level = test["true_level_mm"].to_numpy(np.float32)

    lower = mean_level - half_width
    upper = mean_level + half_width
    covered = (true_level >= lower) & (true_level <= upper)
    metrics = point_metrics(true_level, mean_level, true_delta, mean_delta)
    metrics.update({
        "nominal_coverage": float(args.coverage),
        "empirical_coverage": float(covered.mean()),
        "mean_half_width_mm": half_width,
    })

    audit = {
        "method": "validation-calibrated symmetric MC-mean residual interval",
        "no_test_label_calibration": True,
        "mc_samples": int(args.mc_samples),
        "calibration_samples": int(args.calibration_samples),
        "calibration_seed": int(args.calibration_seed),
        "test_samples": int(len(test)),
        "horizon_days": int(cfg.horizon_days),
        "grid_anchor": cadence["grid_anchor"],
        **metrics,
    }
    (args.output_dir / "mc_dropout_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    pd.DataFrame({
        "point_index": test_p,
        "target_epoch_index": test_t,
        "target_date": grid[test_t].strftime("%Y-%m-%d"),
        "true_level_mm": true_level,
        "mc_mean_level_mm": mean_level,
        "lower_bound_mm": lower,
        "upper_bound_mm": upper,
        "covered": covered.astype(np.int8),
    }).to_csv(args.output_dir / "mc_dropout_test_intervals.csv", index=False)

    print("\nMC DROPOUT RESULTS")
    print(json.dumps(audit, indent=2))
    print(f"[SAVED] {args.output_dir / 'mc_dropout_audit.json'}")
    print(f"[SAVED] {args.output_dir / 'mc_dropout_test_intervals.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
APLAND_fixed12_ablation_multiseed.py

Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

Reviewer-ready fixed-12-day, multi-seed ablation study for APLAND.

Key guarantees
--------------
* Raw mixed 6/12/24-day acquisitions are reindexed to one fixed 12-day grid.
* Forward filling is used for inputs only.
* A target is eligible only when both t and t-1 are raw observations.
* Correct recent-biased sampling uses 1-(1-u)**p (p>1 -> recent times).
* Every variant and seed uses the same 200,000 test point/time pairs.
* Test sampling never changes for the uniform-training ablation (A8).
* A7 changes bidirectionality only; CNN and attention remain present.
* Results are reported in increment and reconstructed-level spaces.
* Three seeds (42, 123, 2024) are run by default and summarized as mean +/- SD.
* Completed (variant, seed) runs are resumed automatically.

Example
-------
python APLAND_fixed12_ablation_multiseed.py \
  --csv "data/EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv" \
  --output-dir "outputs/fixed12_ablation"

Run only the five reviewer-priority configurations first:
python APLAND_fixed12_ablation_multiseed.py --csv "data/EGMS_L2b.csv" \
  --variants A1,A2,A5,A6,A9

Later run the remaining variants using the same output directory; resume logic
will preserve prior results:
python APLAND_fixed12_ablation_multiseed.py --csv "data/EGMS_L2b.csv" \
  --variants A3,A4,A7,A8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, losses, models, optimizers


LOOKBACK = 24
PRED_STEPS = 1
HORIZON_DAYS = 12
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
EPOCHS = 60
BATCH_SIZE = 1024
STEPS_PER_EPOCH = 600
VALIDATION_STEPS = 120
TEST_SAMPLE_N = 200_000
BIAS_POWER = 2.0
TEST_SAMPLE_SEED = 42
SEEDS = (42, 123, 2024)
DATE_RE = re.compile(r"^\d{8}$")


VARIANTS = {
    "A1": ("Full model", {}),
    "A2": ("No delta mode", {"target_mode": "level"}),
    "A3": ("No detrending", {"detrend": False}),
    "A4": ("No slope channel", {"use_slope": False}),
    "A5": ("No attention", {"use_attention": False}),
    "A6": ("No winsorization", {"winsor": None}),
    "A7": ("Unidirectional LSTM", {"bidirectional": False}),
    "A8": ("Uniform temporal sampling", {"recent_sampling": False}),
    "A9": ("No label clipping", {"label_clip_sigma": None}),
}

DEFAULT_CFG = {
    "target_mode": "delta",
    "detrend": True,
    "use_slope": True,
    "use_attention": True,
    "winsor": 6.0,
    "bidirectional": True,
    "recent_sampling": True,
    "label_clip_sigma": 3.0,
}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/EGMS_L2b.csv"),
        help=(
            "Path to the EGMS L2b CSV file. Relative paths are resolved from "
            "the current working directory (default: data/EGMS_L2b.csv)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fixed12_ablation"),
        help=(
            "Directory for checkpoints, audits, tables, and plots "
            "(default: outputs/fixed12_ablation)."
        ),
    )
    p.add_argument("--anchor", default=None, help="YYYYMMDD; default first date")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p.add_argument("--test-samples", type=int, default=TEST_SAMPLE_N)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--steps-per-epoch", type=int, default=STEPS_PER_EPOCH)
    p.add_argument("--validation-steps", type=int, default=VALIDATION_STEPS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    return p.parse_args()


def parse_selection(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    selected = [x.strip().upper() for x in args.variants.split(",") if x.strip()]
    unknown = sorted(set(selected) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return selected, seeds


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def date_columns(columns: pd.Index) -> list[str]:
    cols = [str(c) for c in columns if DATE_RE.fullmatch(str(c))]
    if not cols:
        raise ValueError("No YYYYMMDD columns found.")
    parsed = pd.to_datetime(cols, format="%Y%m%d", errors="raise")
    return [cols[i] for i in np.argsort(parsed.values)]


def regrid_fixed12(
    df: pd.DataFrame, anchor_text: str | None, out_dir: Path
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
    cols = date_columns(df.columns)
    raw_dates = pd.DatetimeIndex(pd.to_datetime(cols, format="%Y%m%d"))
    deltas = np.diff(raw_dates.values).astype("timedelta64[D]").astype(int)
    anchor = (
        pd.to_datetime(anchor_text, format="%Y%m%d")
        if anchor_text
        else raw_dates.min()
    )
    if anchor not in raw_dates:
        raise ValueError(f"Anchor {anchor.date()} is not an acquisition date.")
    count = int((raw_dates.max() - anchor).days // HORIZON_DAYS) + 1
    grid = pd.DatetimeIndex(
        [anchor + pd.Timedelta(days=HORIZON_DAYS * i) for i in range(count)]
    )
    observed = np.asarray([d in set(raw_dates) for d in grid], dtype=bool)
    temporal = df[cols].copy()
    temporal.columns = raw_dates
    fixed = temporal.reindex(columns=grid)
    missing_cells = int(fixed.isna().to_numpy().sum())
    fixed_clean = fixed.ffill(axis=1).fillna(0.0).astype(np.float32)

    transition_ok = observed.copy()
    transition_ok[0] = False
    transition_ok[1:] &= observed[:-1]
    excluded_targets = [
        grid[i].strftime("%Y%m%d")
        for i in range(1, len(grid))
        if not transition_ok[i]
    ]
    cadence = {
        "raw_epoch_count": len(raw_dates),
        "raw_interval_frequency": {
            str(k): int(np.sum(deltas == k)) for k in sorted(set(deltas))
        },
        "fixed_grid_epoch_count": len(grid),
        "observed_grid_epochs": int(observed.sum()),
        "absent_grid_dates": [grid[i].strftime("%Y%m%d") for i in np.where(~observed)[0]],
        "excluded_target_dates": excluded_targets,
        "missing_cells_before_imputation": missing_cells,
        "missing_rate_before_imputation": missing_cells / fixed.size,
    }
    (out_dir / "cadence_and_target_audit.json").write_text(
        json.dumps(cadence, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "epoch_index": np.arange(len(grid)),
            "date": grid.strftime("%Y-%m-%d"),
            "raw_observed": observed,
            "eligible_transition": transition_ok,
        }
    ).to_csv(out_dir / "fixed12_grid_and_targets.csv", index=False)
    return fixed_clean.to_numpy(np.float32), grid, observed


def eligible_times(observed: np.ndarray, start: int, stop: int) -> np.ndarray:
    transition_ok = observed.copy()
    transition_ok[0] = False
    transition_ok[1:] &= observed[:-1]
    idx = np.arange(max(start, LOOKBACK), stop, dtype=np.int64)
    return idx[transition_ok[idx]]


def sample_ranks(
    rng: np.random.Generator, n_choices: int, size: int, recent: bool
) -> np.ndarray:
    if recent:
        u = rng.random(size)
        v = 1.0 - np.power(1.0 - u, BIAS_POWER)
        ranks = np.floor(n_choices * v).astype(np.int64)
        return np.minimum(ranks, n_choices - 1)
    return rng.integers(0, n_choices, size=size)


def sample_times(
    rng: np.random.Generator, valid_times: np.ndarray, size: int, recent: bool
) -> np.ndarray:
    if valid_times.size == 0:
        raise ValueError("No eligible target epochs in split.")
    return valid_times[sample_ranks(rng, valid_times.size, size, recent)]


def build_features(window: np.ndarray, cfg: dict) -> np.ndarray:
    w = window.astype(np.float32).copy()
    if cfg["detrend"]:
        t = np.arange(LOOKBACK, dtype=np.float32)
        A = np.column_stack((t, np.ones_like(t)))
        slope, intercept = np.linalg.lstsq(A, w, rcond=None)[0]
        w0 = w - (slope * t + intercept)
    else:
        w0 = w
    channels = [w0]
    if cfg["use_slope"]:
        channels.append(np.diff(w0, prepend=w0[0]))
    X = np.stack(channels, axis=-1)
    if cfg["winsor"] is not None:
        X = np.clip(X, -float(cfg["winsor"]), float(cfg["winsor"]))
    return X.astype(np.float32)


def batch(
    arr_norm: np.ndarray,
    valid_times: np.ndarray,
    batch_size: int,
    cfg: dict,
    rng: np.random.Generator,
    clip_labels: bool,
) -> tuple[np.ndarray, np.ndarray]:
    times = sample_times(rng, valid_times, batch_size, cfg["recent_sampling"])
    points = rng.integers(0, arr_norm.shape[0], size=batch_size)
    channels = 1 + int(cfg["use_slope"])
    X = np.empty((batch_size, LOOKBACK, channels), np.float32)
    y = np.empty((batch_size, 1), np.float32)
    for i, (point, target) in enumerate(zip(points, times)):
        X[i] = build_features(arr_norm[point, target - LOOKBACK : target], cfg)
        if cfg["target_mode"] == "delta":
            y[i, 0] = arr_norm[point, target] - arr_norm[point, target - 1]
        else:
            y[i, 0] = arr_norm[point, target]
    if clip_labels and cfg["label_clip_sigma"] is not None:
        limit = max(0.5, float(cfg["label_clip_sigma"]) * (float(np.std(y)) + 1e-8))
        y = np.clip(y, -limit, limit)
    return X, y


def dataset(
    arr_norm: np.ndarray,
    valid_times: np.ndarray,
    batch_size: int,
    cfg: dict,
    seed: int,
) -> tf.data.Dataset:
    channels = 1 + int(cfg["use_slope"])

    def generator():
        rng = np.random.default_rng(seed)
        while True:
            yield batch(arr_norm, valid_times, batch_size, cfg, rng, True)

    signature = (
        tf.TensorSpec((None, LOOKBACK, channels), tf.float32),
        tf.TensorSpec((None, 1), tf.float32),
    )
    return tf.data.Dataset.from_generator(generator, output_signature=signature).repeat().prefetch(2)


def shared_test_indices(
    valid_test_times: np.ndarray, n_points: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(TEST_SAMPLE_SEED)
    # Test protocol is fixed for every variant: corrected recent-biased p=2.
    times = sample_times(rng, valid_test_times, n, recent=True)
    points = rng.integers(0, n_points, size=n)
    return points, times


def test_arrays(
    arr_norm: np.ndarray,
    points: np.ndarray,
    times: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    channels = 1 + int(cfg["use_slope"])
    X = np.empty((len(points), LOOKBACK, channels), np.float32)
    y = np.empty((len(points), 1), np.float32)
    for i, (point, target) in enumerate(zip(points, times)):
        X[i] = build_features(arr_norm[point, target - LOOKBACK : target], cfg)
        if cfg["target_mode"] == "delta":
            y[i, 0] = arr_norm[point, target] - arr_norm[point, target - 1]
        else:
            y[i, 0] = arr_norm[point, target]
    return X, y


def recurrent_layer(units: int, cfg: dict, return_sequences: bool):
    cell = layers.LSTM(
        units, return_sequences=return_sequences, dropout=0.25, recurrent_dropout=0.1
    )
    return layers.Bidirectional(cell) if cfg["bidirectional"] else cell


def build_model(cfg: dict, steps_per_epoch: int) -> tf.keras.Model:
    channels = 1 + int(cfg["use_slope"])
    inputs = layers.Input((LOOKBACK, channels))
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inputs)
    x = layers.Conv1D(64, 5, padding="causal", activation="gelu")(x)
    x = layers.LayerNormalization()(x)

    if cfg["use_attention"]:
        x = recurrent_layer(128, cfg, True)(x)
        x = recurrent_layer(64, cfg, True)(x)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
    else:
        x = recurrent_layer(128, cfg, True)(x)
        x = recurrent_layer(64, cfg, False)(x)
        x = layers.LayerNormalization()(x)

    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    output = layers.Dense(1)(x)
    model = models.Model(inputs, output)
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=max(1, steps_per_epoch * 5)
    )
    model.compile(
        optimizer=optimizers.AdamW(schedule, weight_decay=1e-4, clipnorm=1.0),
        loss=losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = float(np.sum(np.square(y_true - np.mean(y_true)), dtype=np.float64))
    numerator = float(np.sum(np.square(error), dtype=np.float64))
    r2 = float(1.0 - numerator / denominator) if denominator > 0 else math.nan
    return mae, rmse, r2


def evaluate(
    model: tf.keras.Model,
    X: np.ndarray,
    y_norm: np.ndarray,
    points: np.ndarray,
    times: np.ndarray,
    arr_values: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    cfg: dict,
) -> dict:
    pred_norm = model.predict(X, batch_size=4096, verbose=1).reshape(-1)
    point_std = stds[points, 0]
    point_mean = means[points, 0]
    last_level = arr_values[points, times - 1]
    true_level = arr_values[points, times]
    true_delta = true_level - last_level
    if cfg["target_mode"] == "delta":
        pred_delta = pred_norm * point_std
        pred_level = last_level + pred_delta
    else:
        pred_level = pred_norm * point_std + point_mean
        pred_delta = pred_level - last_level
    delta = metric_values(true_delta, pred_delta)
    level = metric_values(true_level, pred_level)
    return {
        "MAE_delta_mm": delta[0],
        "RMSE_delta_mm": delta[1],
        "R2_delta": delta[2],
        "MAE_level_mm": level[0],
        "RMSE_level_mm": level[1],
        "R2_level": level[2],
    }


def summarize(raw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    metrics = [
        "MAE_delta_mm", "RMSE_delta_mm", "R2_delta",
        "MAE_level_mm", "RMSE_level_mm", "R2_level", "Train_minutes",
    ]
    ok = raw[raw["Status"] == "OK"].copy()
    grouped = ok.groupby(["ID", "Variant"], sort=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{a}_{b}" for a, b in grouped.columns]
    summary = grouped.reset_index()
    full = summary.loc[summary["ID"] == "A1"]
    if not full.empty:
        for metric in ("MAE_delta_mm", "RMSE_delta_mm", "R2_delta"):
            baseline = float(full[f"{metric}_mean"].iloc[0])
            summary[f"Delta_{metric}_vs_A1"] = summary[f"{metric}_mean"] - baseline
    summary.to_csv(out_dir / "ablation_summary_mean_sd.csv", index=False)
    return summary


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    cols = [
        ("MAE_delta_mm", "Increment MAE (mm)", False),
        ("RMSE_delta_mm", "Increment RMSE (mm)", False),
        ("R2_delta", "Increment R-squared", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, title, _) in zip(axes, cols):
        means = summary[f"{metric}_mean"].to_numpy(float)
        errors = summary[f"{metric}_std"].fillna(0).to_numpy(float)
        colors = ["#2E8B57" if x == "A1" else "#B23A48" for x in summary["ID"]]
        ax.bar(summary["ID"], means, yerr=errors, capsize=4, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Fixed 12-day multi-seed ablation (mean +/- SD)")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_summary_mean_sd.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = arguments()
    selected, seeds = parse_selection(args)
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Place the EGMS CSV in data/EGMS_L2b.csv or provide its location "
            "with --csv."
        )
    if not csv_path.is_file():
        raise ValueError(f"Input CSV path is not a file: {csv_path}")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "ablation_log.txt", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("ablation")
    run_cfg = {
        **vars(args),
        "csv": str(csv_path),
        "output_dir": str(out_dir),
        "selected_variants": selected,
        "parsed_seeds": seeds,
    }
    (out_dir / "ablation_run_config.json").write_text(
        json.dumps(run_cfg, indent=2), encoding="utf-8"
    )

    log.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    arr_values, grid, observed = regrid_fixed12(df, args.anchor, out_dir)
    del df
    gc.collect()

    T = arr_values.shape[1]
    train_end = int(T * TRAIN_FRACTION)
    validation_end = int(T * (TRAIN_FRACTION + VALIDATION_FRACTION))
    train_times = eligible_times(observed, LOOKBACK, train_end)
    validation_times = eligible_times(observed, train_end, validation_end)
    test_times = eligible_times(observed, validation_end, T)
    target_audit = {
        "train_count": int(train_times.size),
        "validation_count": int(validation_times.size),
        "test_count": int(test_times.size),
        "train_dates": [grid[i].strftime("%Y-%m-%d") for i in train_times],
        "validation_dates": [grid[i].strftime("%Y-%m-%d") for i in validation_times],
        "test_dates": [grid[i].strftime("%Y-%m-%d") for i in test_times],
    }
    (out_dir / "eligible_target_audit.json").write_text(
        json.dumps(target_audit, indent=2), encoding="utf-8"
    )
    log.info("Eligible target epochs train/val/test: %d/%d/%d", len(train_times), len(validation_times), len(test_times))

    train_values = arr_values[:, :train_end]
    means = np.mean(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    stds = np.std(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    means[~np.isfinite(means)] = 0.0
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    arr_norm = ((arr_values - means) / stds).astype(np.float32)
    shared_points, shared_times = shared_test_indices(
        test_times, arr_values.shape[0], args.test_samples
    )
    pd.DataFrame(
        {"point_index": shared_points, "target_epoch_index": shared_times,
         "target_date": grid[shared_times].strftime("%Y-%m-%d"),
         "horizon_days": (grid[shared_times] - grid[shared_times - 1]).days}
    ).to_csv(out_dir / "shared_test_indices.csv", index=False)

    raw_path = out_dir / "ablation_runs_raw.csv"
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
    else:
        raw = pd.DataFrame()

    for variant_id in selected:
        label, overrides = VARIANTS[variant_id]
        cfg = DEFAULT_CFG | overrides
        for seed in seeds:
            if not raw.empty and ((raw["ID"] == variant_id) & (raw["Seed"] == seed) & (raw["Status"] == "OK")).any():
                log.info("Skip completed %s seed %d", variant_id, seed)
                continue
            log.info("START %s %s seed=%d overrides=%s", variant_id, label, seed, overrides)
            seed_everything(seed)
            tf.keras.backend.clear_session()
            train_ds = dataset(arr_norm, train_times, args.batch_size, cfg, seed)
            val_ds = dataset(arr_norm, validation_times, args.batch_size, cfg, seed + 10_000)
            X_test, y_test = test_arrays(arr_norm, shared_points, shared_times, cfg)
            model = build_model(cfg, args.steps_per_epoch)
            checkpoint = out_dir / f"best_{variant_id}_seed{seed}.keras"
            cbs = [
                callbacks.ModelCheckpoint(checkpoint, monitor="val_loss", save_best_only=True),
                callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                callbacks.CSVLogger(out_dir / f"history_{variant_id}_seed{seed}.csv"),
                callbacks.TerminateOnNaN(),
            ]
            started = time.time()
            try:
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=args.epochs,
                    steps_per_epoch=args.steps_per_epoch,
                    validation_steps=args.validation_steps,
                    callbacks=cbs,
                    verbose=1,
                )
                result = evaluate(
                    model, X_test, y_test, shared_points, shared_times,
                    arr_values, means, stds, cfg,
                )
                row = {
                    "ID": variant_id, "Variant": label, "Seed": seed,
                    **result, "Best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                    "Epochs_run": len(history.history["loss"]),
                    "Train_minutes": (time.time() - started) / 60, "Status": "OK",
                }
                log.info("DONE %s seed=%d delta MAE=%.4f RMSE=%.4f R2=%.4f", variant_id, seed, result["MAE_delta_mm"], result["RMSE_delta_mm"], result["R2_delta"])
            except Exception as exc:
                log.exception("FAILED %s seed=%d", variant_id, seed)
                row = {
                    "ID": variant_id, "Variant": label, "Seed": seed,
                    "Status": "FAILED", "Error": str(exc),
                    "Train_minutes": (time.time() - started) / 60,
                }
            raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
            raw.to_csv(raw_path, index=False)
            del model, X_test, y_test, train_ds, val_ds
            tf.keras.backend.clear_session()
            gc.collect()

    summary = summarize(raw, out_dir)
    plot_summary(summary, out_dir)
    log.info("FINAL SUMMARY\n%s", summary.to_string(index=False))
    log.info("Ablation study complete: %s", out_dir)


if __name__ == "__main__":
    main()

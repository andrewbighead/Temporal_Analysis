#!/usr/bin/env python3
"""Limited, validation-only hyperparameter search for the fixed-12-day model.

Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

This script reuses the functions in APLAND_fixed12_training.py and runs the
seven prespecified configurations requested for the revision. Hyperparameters
are selected exclusively on a fixed validation sample; the held-out test set is
never constructed or evaluated here.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEARCH_GRID = [
    {"id": "H1", "lookback": 18, "huber_delta": 1.0, "bias_power": 2.0},
    {"id": "H2", "lookback": 24, "huber_delta": 1.0, "bias_power": 2.0},
    {"id": "H3", "lookback": 30, "huber_delta": 1.0, "bias_power": 2.0},
    {"id": "H4", "lookback": 24, "huber_delta": 0.5, "bias_power": 2.0},
    {"id": "H5", "lookback": 24, "huber_delta": 2.0, "bias_power": 2.0},
    {"id": "H6", "lookback": 24, "huber_delta": 1.0, "bias_power": 1.0},
    {"id": "H7", "lookback": 24, "huber_delta": 1.0, "bias_power": 3.0},
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a seven-configuration validation-only search."
    )
    p.add_argument(
        "--csv", type=Path, default=Path("data/EGMS_L2b.csv"),
        help="Raw EGMS CSV file (default: data/EGMS_L2b.csv)",
    )
    p.add_argument(
        "--base-script",
        type=Path,
        default=Path("APLAND_fixed12_training.py"),
        help="Path to the verified fixed-12-day training script",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fixed12_hyperparameter_search"),
        help="Output directory (default: outputs/fixed12_hyperparameter_search)",
    )
    p.add_argument("--anchor", default=None, help="Optional YYYYMMDD grid anchor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=600)
    p.add_argument("--validation-steps", type=int, default=120)
    p.add_argument(
        "--validation-samples",
        type=int,
        default=100_000,
        help="Size of the common fixed validation sample",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun configurations that already have a completed result JSON",
    )
    return p.parse_args()


def load_base_module(script_path: Path):
    if not script_path.exists():
        raise FileNotFoundError(f"Base script not found: {script_path}")
    spec = importlib.util.spec_from_file_location("apland_fixed12_base", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import base script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def make_common_validation_indices(
    valid_transition: np.ndarray,
    train_end: int,
    val_end: int,
    n_points: int,
    n_samples: int,
    seed: int,
):
    # Targets are drawn uniformly from validation epochs. This evaluation draw
    # is independent of BIAS_POWER, ensuring a fair comparison of H1--H7.
    valid_times = np.flatnonzero(valid_transition)
    valid_times = valid_times[(valid_times >= train_end) & (valid_times < val_end)]
    if valid_times.size == 0:
        raise RuntimeError("No eligible target epochs in the validation interval")
    rng = np.random.default_rng(seed)
    points = rng.integers(0, n_points, size=n_samples, dtype=np.int32)
    times = rng.choice(valid_times, size=n_samples, replace=True).astype(np.int32)
    return points, times, valid_times.astype(np.int32)


def build_fixed_validation(base, arr_norm, points, times, cfg):
    n = len(points)
    x = np.empty((n, cfg.lookback, 2), dtype=np.float32)
    y_norm = np.empty(n, dtype=np.float32)
    for i, (point_idx, target_t) in enumerate(zip(points, times)):
        window = arr_norm[point_idx, target_t - cfg.lookback : target_t]
        x[i] = base.detrended_channels(window, cfg)
        y_norm[i] = arr_norm[point_idx, target_t] - arr_norm[point_idx, target_t - 1]
    return x, y_norm


def compile_model(base, cfg):
    model = base.build_model(cfg)
    schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=cfg.learning_rate,
        first_decay_steps=max(1, cfg.steps_per_epoch * 5),
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=schedule,
            weight_decay=cfg.weight_decay,
            clipnorm=1.0,
        ),
        loss=tf.keras.losses.Huber(delta=cfg.huber_delta),
        metrics=["mae"],
    )
    return model


def train_one(
    base,
    cfg,
    run_id,
    run_dir,
    arr_norm,
    stds,
    valid_transition,
    train_end,
    val_end,
    val_points,
    val_times,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    gc.collect()
    base.set_seeds(cfg.seed)

    train_indices = np.arange(cfg.lookback, train_end, dtype=np.int64)
    train_times = train_indices[valid_transition[train_indices]]
    validation_indices = np.arange(max(train_end, cfg.lookback), val_end, dtype=np.int64)
    validation_times = validation_indices[valid_transition[validation_indices]]
    if train_times.size == 0 or validation_times.size == 0:
        raise RuntimeError(f"{run_id}: empty training or validation target set")

    train_ds = base.make_tf_dataset(
        arr_norm, train_times, cfg, cfg.seed, clip_labels=True
    )
    sampled_validation_ds = base.make_tf_dataset(
        arr_norm, validation_times, cfg, cfg.seed + 10_000, clip_labels=True
    )

    model = compile_model(base, cfg)
    checkpoint = run_dir / "best_model.keras"
    history_csv = run_dir / "training_history.csv"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(history_csv)),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=sampled_validation_ds,
        epochs=cfg.epochs,
        steps_per_epoch=cfg.steps_per_epoch,
        validation_steps=cfg.validation_steps,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - started

    # ModelCheckpoint is the authoritative best sampled-validation checkpoint.
    model = tf.keras.models.load_model(checkpoint)
    x_val, y_val_norm = build_fixed_validation(
        base, arr_norm, val_points, val_times, cfg
    )
    pred_norm = model.predict(x_val, batch_size=4096, verbose=1).reshape(-1)
    point_scale = stds[val_points, 0].astype(np.float32)
    true_delta_mm = y_val_norm * point_scale
    pred_delta_mm = pred_norm * point_scale

    val_loss_history = np.asarray(history.history["val_loss"], dtype=float)
    best_epoch = int(np.nanargmin(val_loss_history) + 1)
    result = {
        "id": run_id,
        "status": "completed",
        "lookback": int(cfg.lookback),
        "huber_delta": float(cfg.huber_delta),
        "bias_power": float(cfg.bias_power),
        "seed": int(cfg.seed),
        "epochs_completed": int(len(val_loss_history)),
        "best_epoch": best_epoch,
        "best_sampled_val_loss": float(np.nanmin(val_loss_history)),
        "fixed_validation_samples": int(len(val_points)),
        "fixed_validation_mae_mm": float(mean_absolute_error(true_delta_mm, pred_delta_mm)),
        "fixed_validation_rmse_mm": float(
            np.sqrt(mean_squared_error(true_delta_mm, pred_delta_mm))
        ),
        "fixed_validation_r2_delta": float(r2_score(true_delta_mm, pred_delta_mm)),
        "fixed_validation_mae_normalized": float(
            mean_absolute_error(y_val_norm, pred_norm)
        ),
        "elapsed_minutes": float(elapsed / 60.0),
        "selection_data": "validation only; held-out test set not accessed",
    }
    write_json(run_dir / "result.json", result)
    del model, train_ds, sampled_validation_ds, x_val, pred_norm
    tf.keras.backend.clear_session()
    gc.collect()
    return result


def collect_results(output_dir: Path):
    results = []
    for item in SEARCH_GRID:
        result_path = output_dir / item["id"] / "result.json"
        if result_path.exists():
            with result_path.open("r", encoding="utf-8") as f:
                result = json.load(f)
            if result.get("status") == "completed":
                results.append(result)
    return results


def write_summary(output_dir: Path, results):
    if not results:
        return
    df = pd.DataFrame(results)
    df = df.sort_values("fixed_validation_mae_mm", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df.to_csv(output_dir / "hyperparameter_search_summary.csv", index=False)
    try:
        df.to_excel(output_dir / "hyperparameter_search_summary.xlsx", index=False)
    except Exception as exc:
        print(f"[WARN] Excel export skipped: {exc}")

    selected = df.iloc[0].to_dict()
    selected["selection_rule"] = "minimum fixed-validation MAE (mm)"
    selected["test_set_used_for_selection"] = False
    write_json(output_dir / "selected_configuration.json", selected)

    by_id = df.sort_values("id")
    plt.figure(figsize=(8, 4.8))
    plt.bar(by_id["id"], by_id["fixed_validation_mae_mm"], color="#2878B5")
    plt.ylabel("Fixed-validation MAE (mm)")
    plt.xlabel("Configuration")
    plt.title("Limited hyperparameter search (validation only)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "hyperparameter_search_validation_mae.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    base_script = args.base_script.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {csv_path}\n"
            "Place the file at data/EGMS_L2b.csv or provide --csv."
        )
    if not base_script.is_file():
        raise FileNotFoundError(f"Base training script not found: {base_script}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = load_base_module(base_script)

    protocol = {
        "purpose": "limited hyperparameter search",
        "selection_rule": "minimum MAE on one common fixed validation sample",
        "test_set_used_for_selection": False,
        "search_grid": SEARCH_GRID,
        "seed": args.seed,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_steps": args.validation_steps,
        "fixed_validation_samples": args.validation_samples,
        "base_script": str(base_script),
        "csv": str(csv_path),
    }
    write_json(output_dir / "search_protocol.json", protocol)

    # Load and preprocess once. Every configuration sees the same time split,
    # normalization, eligibility mask, and common fixed validation targets.
    loader_cfg = base.Config(
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_steps=args.validation_steps,
    )
    base.set_seeds(args.seed)
    print(f"[INFO] Reading and constructing fixed 12-day grid: {csv_path}")
    _, arr_values, grid, observed, _ = base.load_and_regrid(
        csv_path, output_dir, loader_cfg, args.anchor
    )
    n_points, n_times = arr_values.shape
    train_end = int(n_times * loader_cfg.train_fraction)
    val_end = int(
        n_times * (loader_cfg.train_fraction + loader_cfg.validation_fraction)
    )
    train_values = arr_values[:, :train_end]
    means = np.mean(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    stds = np.std(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    stds[~np.isfinite(stds) | (stds == 0)] = 1.0
    means[~np.isfinite(means)] = 0.0
    arr_norm = ((arr_values - means) / stds).astype(np.float32)
    valid_transition = observed.copy()
    valid_transition[0] = False
    valid_transition[1:] &= observed[:-1]

    val_points, val_times, validation_epochs = make_common_validation_indices(
        valid_transition,
        train_end,
        val_end,
        n_points,
        args.validation_samples,
        args.seed + 2026,
    )
    np.savez_compressed(
        output_dir / "fixed_validation_sample.npz",
        point_index=val_points,
        target_time_index=val_times,
        target_dates=np.asarray([str(grid[t].date()) for t in val_times]),
    )
    write_json(
        output_dir / "validation_audit.json",
        {
            "sample_count": int(len(val_points)),
            "sampling": "uniform over eligible validation target epochs",
            "sampling_seed": int(args.seed + 2026),
            "eligible_validation_epoch_count": int(len(validation_epochs)),
            "first_validation_target_date": str(grid[validation_epochs.min()].date()),
            "last_validation_target_date": str(grid[validation_epochs.max()].date()),
            "test_set_accessed": False,
        },
    )

    for hp in SEARCH_GRID:
        run_id = hp["id"]
        run_dir = output_dir / run_id
        result_path = run_dir / "result.json"
        if result_path.exists() and not args.force:
            try:
                with result_path.open("r", encoding="utf-8") as f:
                    old = json.load(f)
                if old.get("status") == "completed":
                    print(f"[SKIP] {run_id} already completed. Use --force to rerun.")
                    write_summary(output_dir, collect_results(output_dir))
                    continue
            except Exception:
                pass

        cfg = base.Config(
            seed=args.seed,
            lookback=hp["lookback"],
            bias_power=hp["bias_power"],
            huber_delta=hp["huber_delta"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure_path.unlink()
        write_json(run_dir / "config.json", asdict(cfg))
        print("\n" + "=" * 78)
        print(
            f"[RUN {run_id}] LOOKBACK={cfg.lookback}, "
            f"Huber delta={cfg.huber_delta}, BIAS_POWER={cfg.bias_power}"
        )
        print("=" * 78)
        try:
            result = train_one(
                base,
                cfg,
                run_id,
                run_dir,
                arr_norm,
                stds,
                valid_transition,
                train_end,
                val_end,
                val_points,
                val_times,
            )
            if failure_path.exists():
                failure_path.unlink()
            print(
                f"[DONE {run_id}] fixed-validation MAE="
                f"{result['fixed_validation_mae_mm']:.4f} mm"
            )
        except KeyboardInterrupt:
            print("[STOP] Interrupted by user. Completed runs are resumable.")
            write_summary(output_dir, collect_results(output_dir))
            raise
        except Exception as exc:
            failure = {
                "id": run_id,
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(run_dir / "failure.json", failure)
            print(f"[ERROR {run_id}] {exc}")
        finally:
            write_summary(output_dir, collect_results(output_dir))

    results = collect_results(output_dir)
    write_summary(output_dir, results)
    if len(results) == len(SEARCH_GRID):
        best = min(results, key=lambda r: r["fixed_validation_mae_mm"])
        print("\n[SEARCH COMPLETE]")
        print(
            f"Selected {best['id']}: LOOKBACK={best['lookback']}, "
            f"Huber delta={best['huber_delta']}, BIAS_POWER={best['bias_power']}, "
            f"validation MAE={best['fixed_validation_mae_mm']:.4f} mm"
        )
        print("The held-out test set was not used or evaluated in this search.")
    else:
        print(f"[PARTIAL] {len(results)}/{len(SEARCH_GRID)} runs completed; rerun to resume.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
APLAND_fixed12_cluster_stratification.py

Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

Cluster-stratified evaluation on the exact saved fixed-12-day test sample.

Combines ``test_predictions_fixed12.csv`` from the final model with the one
unified ``cluster_labels_all.npy``. It never redraws test windows or reruns the
model. XAI is intentionally handled by the dedicated coherent-XAI script.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "point_index", "target_epoch_index", "target_date", "horizon_days",
    "true_increment_mm", "predicted_increment_mm", "true_level_mm",
    "predicted_level_mm",
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/fixed12_training/test_predictions_fixed12.csv"),
        help="Saved predictions for the fixed test sample.",
    )
    p.add_argument(
        "--labels", type=Path,
        default=Path("outputs/fixed12_clustering/cluster_labels_all.npy"),
        help="Unified PCA--KMeans label array for all spatial points.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/fixed12_cluster_stratification"),
    )
    p.add_argument("--expected-samples", type=int, default=200_000)
    p.add_argument("--expected-points", type=int, default=594_978)
    p.add_argument("--expected-clusters", type=int, default=4)
    return p.parse_args()


def metric_values(true, pred):
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    error = pred - true
    denominator = np.sum((true - np.mean(true)) ** 2, dtype=np.float64)
    r2 = 1.0 - np.sum(error ** 2, dtype=np.float64) / denominator if denominator > 0 else math.nan
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2))), float(r2)


def main():
    args = arguments()
    pred_path = args.predictions.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction table not found: {pred_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Cluster-label array not found: {labels_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path, low_memory=False)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    if len(df) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples:,} rows, found {len(df):,}")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Required prediction columns contain missing values")
    horizons = sorted(df["horizon_days"].astype(int).unique().tolist())
    if horizons != [12]:
        raise ValueError(f"Prediction table is not exclusively 12-day-ahead: {horizons}")

    labels = np.load(labels_path, allow_pickle=False)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Cluster labels must be a one-dimensional integer array")
    if len(labels) != args.expected_points:
        raise ValueError(
            f"Expected {args.expected_points:,} point labels, found {len(labels):,}"
        )
    points = df["point_index"].to_numpy(np.int64)
    if points.min() < 0 or points.max() >= len(labels):
        raise IndexError("Prediction point_index is outside the cluster-label array")
    window_cluster = labels[points]
    # Cluster IDs (0-3) are assigned by APLAND_fixed12_unified_clusters.py
    # using canonicalize_labels() for stable ordering by net displacement.
    # k=4 was retained for interpretability (silhouette peaks at k=2=0.410;
    # k=4=0.366 yields qualitatively distinct trajectory regimes).
    clusters = sorted(np.unique(labels).astype(int).tolist())
    expected_cluster_ids = list(range(args.expected_clusters))
    if clusters != expected_cluster_ids:
        raise ValueError(
            f"Expected cluster IDs {expected_cluster_ids}; found {clusters}"
        )

    rows, epoch_rows = [], []
    for cluster in clusters:
        mask = window_cluster == cluster
        if not np.any(mask):
            raise ValueError(f"Cluster {cluster} has no windows in the fixed test sample")
        true_delta = df.loc[mask, "true_increment_mm"].to_numpy(float)
        pred_delta = df.loc[mask, "predicted_increment_mm"].to_numpy(float)
        true_level = df.loc[mask, "true_level_mm"].to_numpy(float)
        pred_level = df.loc[mask, "predicted_level_mm"].to_numpy(float)
        mae_d, rmse_d, r2_d = metric_values(true_delta, pred_delta)
        mae_l, rmse_l, r2_l = metric_values(true_level, pred_level)
        rows.append({
            "Cluster": cluster,
            "n_points_full_dataset": int(np.sum(labels == cluster)),
            "n_test_windows": int(mask.sum()),
            "n_unique_test_points": int(df.loc[mask, "point_index"].nunique()),
            "n_test_target_epochs": int(df.loc[mask, "target_epoch_index"].nunique()),
            "MAE_delta_mm": mae_d, "RMSE_delta_mm": rmse_d, "R2_delta": r2_d,
            "MAE_level_mm": mae_l, "RMSE_level_mm": rmse_l, "R2_level": r2_l,
        })
        losses = pd.DataFrame({
            "Cluster": cluster,
            "target_epoch_index": df.loc[mask, "target_epoch_index"].astype(int),
            "target_date": df.loc[mask, "target_date"],
            "absolute_error_mm": np.abs(pred_delta - true_delta),
            "squared_error_mm2": (pred_delta - true_delta) ** 2,
        })
        epoch_rows.append(losses.groupby(
            ["Cluster", "target_epoch_index", "target_date"], sort=True
        ).mean().reset_index())

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "cluster_stratified_metrics_fixed12.csv", index=False)
    pd.concat(epoch_rows, ignore_index=True).to_csv(
        out_dir / "cluster_epoch_losses_fixed12.csv", index=False)
    distribution = table[[
        "Cluster", "n_points_full_dataset", "n_test_windows", "n_unique_test_points"
    ]].copy()
    distribution["full_dataset_pct"] = 100 * distribution["n_points_full_dataset"] / len(labels)
    distribution["test_window_pct"] = 100 * distribution["n_test_windows"] / len(df)
    distribution.to_csv(out_dir / "cluster_sample_distribution.csv", index=False)

    latex = [
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"Cluster & Points & Test windows & MAE$_\Delta$ & RMSE$_\Delta$ & $R^2_\Delta$ & MAE$_L$ & RMSE$_L$ & $R^2_L$ \\",
        r"\midrule",
    ]
    for row in rows:
        latex.append(
            f"C{row['Cluster']} & {row['n_points_full_dataset']:,} & {row['n_test_windows']:,} & "
            f"{row['MAE_delta_mm']:.3f} & {row['RMSE_delta_mm']:.3f} & {row['R2_delta']:.3f} & "
            f"{row['MAE_level_mm']:.3f} & {row['RMSE_level_mm']:.3f} & {row['R2_level']:.3f} \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "cluster_stratified_metrics_fixed12.tex").write_text(
        "\n".join(latex), encoding="utf-8")

    audit = {
        "predictions_file": str(pred_path), "labels_file": str(labels_path),
        "test_sample_count": len(df), "full_point_count": len(labels),
        "cluster_ids": clusters, "horizon_days": horizons,
        "sample_action": "none; filtered the exact saved 200k test rows",
        "cluster_assignment_rule": "cluster_labels_all[point_index]",
        "semantic_cluster_names": (
            "not assigned automatically; inspect cluster_profiles.csv before naming regimes"
        ),
    }
    (out_dir / "cluster_stratification_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")
    print("\nFIXED-12-DAY CLUSTER-STRATIFIED PERFORMANCE")
    print(table.to_string(index=False))
    print(f"\n[DONE] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()

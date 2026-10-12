#!/usr/bin/env python3
"""
Paper: "Machine Learning and Explainable Deep Learning for Forecasting Ground
        Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

APLAND_fixed12_unified_clusters.py
===================================
Creates one reproducible PCA--KMeans assignment for all EGMS points. PCA is
fit on a fixed random sample of at most 150,000 standardized trajectories,
all trajectories are projected through that fitted PCA, and one KMeans model
is fit on the complete projected dataset. The saved labels are the single
source of cluster membership for forecasting and XAI stratification.

Output:
  cluster_labels_all.npy   — shape (594978,), dtype int32
  cluster_counts.csv       — per-cluster point counts
  cluster_profiles.csv     — interpretable trajectory summaries
  fixed12_cluster_audit.json — cadence and reproducibility record

Example
-------
python APLAND_fixed12_unified_clusters.py
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

KMEANS_CLUSTERS = 4   # retained for interpretability (4 physically distinct regimes);
                      # silhouette peaks at k=2 (0.410) but k=4 (0.366) yields
                      # quasi-stable / oscillatory / progressive / transitional clusters
PCA_SAMPLE      = 150_000
RANDOM_STATE    = 0
HORIZON_DAYS    = 12

def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("data/EGMS_L2b.csv"))
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs/fixed12_clusters")
    )
    p.add_argument("--anchor", default=None,
               help="YYYYMMDD grid anchor. If omitted, defaults to the earliest "
                    "acquisition. Must match the anchor used in APLAND_fixed12_training.py "
                    "(see cadence_summary.json: grid_anchor).")
    p.add_argument("--pca-sample", type=int, default=PCA_SAMPLE)
    p.add_argument("--clusters", type=int, default=KMEANS_CLUSTERS)
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p.parse_args()

def fixed12_values(df: pd.DataFrame, anchor_text: str | None = None):
    cols = [str(c) for c in df.columns if _is_date_col(c)]
    if not cols:
        raise ValueError("No YYYYMMDD columns found")
    dates = pd.DatetimeIndex(pd.to_datetime(cols, format="%Y%m%d"))
    order = np.argsort(dates.values)
    dates, cols = dates[order], [cols[i] for i in order]
    if dates.has_duplicates:
        raise ValueError("Duplicate acquisition-date columns were detected.")
    deltas = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    anchor = pd.to_datetime(anchor_text, format="%Y%m%d") if anchor_text else dates[0]
    if anchor not in dates:
        raise ValueError(f"Anchor {anchor.date()} is not a raw acquisition date")
    count = int((dates[-1] - anchor).days // HORIZON_DAYS) + 1
    grid = pd.date_range(anchor, periods=count, freq=f"{HORIZON_DAYS}D")
    temporal = df[cols].copy(); temporal.columns = dates
    fixed = temporal.reindex(columns=grid)
    observed = np.asarray([d in set(dates) for d in grid], bool)
    values = fixed.ffill(axis=1).fillna(0.0).to_numpy(np.float32)
    audit = {
        "raw_epoch_count": len(dates),
        "raw_interval_frequency": {str(k): int(np.sum(deltas == k)) for k in sorted(set(deltas))},
        "fixed_grid_epoch_count": len(grid), "grid_anchor": str(anchor.date()),
        "grid_last_date": str(grid[-1].date()), "observed_grid_epochs": int(observed.sum()),
        "absent_grid_dates": [grid[i].strftime("%Y%m%d") for i in np.where(~observed)[0]],
        "imputation_policy": "causal forward-fill for clustering inputs only",
    }
    return values, grid, observed, audit

def canonicalize_labels(labels: np.ndarray, values: np.ndarray, k: int):
    """Give stable IDs by ascending median net displacement, then variability."""
    profiles = []
    for old in range(k):
        mask = labels == old
        net = values[mask, -1] - values[mask, 0]
        variability = np.std(values[mask], axis=1)
        profiles.append((old, float(np.median(net)), float(np.median(variability))))
    order = sorted(profiles, key=lambda x: (x[1], x[2]))
    mapping = {old: new for new, (old, _, _) in enumerate(order)}
    return np.asarray([mapping[int(x)] for x in labels], np.int32), mapping

def main() -> None:
    args = arguments()
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2.")
    if args.pca_sample < args.clusters:
        raise ValueError("--pca-sample must be at least as large as --clusters.")
    csv_path = args.csv.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    wanted = {"pid", "point_id", "id", "longitude", "latitude", "lon", "lat", "easting", "northing"}
    metadata_cols = [c for c in df.columns if str(c).lower() in wanted and not _is_date_col(c)]
    assignment_table = df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=df.index)
    arr_vals, grid, observed, audit = fixed12_values(df, args.anchor)
    del df
    n_points = arr_vals.shape[0]
    row_mean = arr_vals.mean(axis=1, keepdims=True)
    row_std = arr_vals.std(axis=1, keepdims=True)
    row_std[row_std < 1e-8] = 1.0
    arr_std = ((arr_vals - row_mean) / row_std).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(n_points, min(args.pca_sample, n_points), replace=False)
    print(f"[INFO] PCA fit sample: {len(sel):,}; components=2")
    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(arr_std[sel])
    pc_all = pca.transform(arr_std).astype(np.float32)
    print(f"[INFO] KMeans fit on ALL {n_points:,} PCA projections")
    km = KMeans(n_clusters=args.clusters, n_init=20, random_state=args.seed)
    raw_labels = km.fit_predict(pc_all)
    labels_all, mapping = canonicalize_labels(raw_labels, arr_vals, args.clusters)
    joblib.dump(pca, out_dir / "pca_model.joblib")
    joblib.dump(km, out_dir / "kmeans_model_raw_labels.joblib")
    np.save(out_dir / "cluster_labels_all.npy", labels_all)
    np.save(out_dir / "pca_coordinates_all.npy", pc_all)
    assignment_table.insert(0, "row_index", np.arange(n_points, dtype=np.int64))
    assignment_table["Cluster"] = labels_all
    assignment_table.to_csv(out_dir / "cluster_assignments_all.csv", index=False)

    rows = []
    for c in range(args.clusters):
        mask = labels_all == c
        net = arr_vals[mask, -1] - arr_vals[mask, 0]
        rows.append({"Cluster": c, "n_points": int(mask.sum()),
                     "pct": 100.0 * mask.mean(),
                     "median_net_displacement_mm": float(np.median(net)),
                     "mean_net_displacement_mm": float(np.mean(net)),
                     "median_temporal_sd_mm": float(np.median(np.std(arr_vals[mask], axis=1)))})
    profiles = pd.DataFrame(rows)
    profiles[["Cluster", "n_points", "pct"]].to_csv(out_dir / "cluster_counts.csv", index=False)
    profiles.to_csv(out_dir / "cluster_profiles.csv", index=False)
    audit.update({"n_points": n_points, "pca_fit_sample": len(sel),
                  "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                  "pca_fit_policy": "fixed random sample without replacement",
                  "kmeans_fit_points": n_points,
                  "kmeans_fit_policy": "fit once on PCA projections of all points",
                  "k": args.clusters, "n_init": 20,
                  "random_state": args.seed,
                  "assignment_metadata_columns": metadata_cols,
                  "canonical_mapping_old_to_new": {str(k): int(v) for k, v in mapping.items()},
                  "canonical_rule": "ascending median net displacement, then median temporal SD",
                  "downstream_rule": "reuse cluster_labels_all.npy; do not refit KMeans"})
    (out_dir / "fixed12_cluster_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    plot_idx = rng.choice(n_points, min(50_000, n_points), replace=False)
    plt.figure(figsize=(7, 5.5))
    plt.scatter(pc_all[plot_idx, 0], pc_all[plot_idx, 1], c=labels_all[plot_idx], s=2, cmap="tab10")
    plt.title(f"Unified fixed-12-day KMeans clusters (k={args.clusters})")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
    plt.savefig(out_dir / "cluster_labels_pca_check.png", dpi=200)
    plt.close()
    print("\nCLUSTER COUNTS — use these labels for BOTH forecasting and XAI")
    print(profiles.to_string(index=False))
    print(f"\n[DONE] Outputs written to {out_dir}")

if __name__ == "__main__":
    main()

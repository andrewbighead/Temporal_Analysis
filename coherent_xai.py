#!/usr/bin/env python3
"""
APLAND_fixed12_coherent_xai.py

Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

Reviewer-ready XAI for the fixed-12-day APLAND forecasting model.

Key safeguards
--------------
* XAI rows are selected from the saved final fixed-12-day test predictions.
* Unified cluster labels are reused; KMeans is never recomputed here.
* Integrated Gradients completeness is measured and saved.
* Coherent deletion/insertion modifies the normalized trajectory first, then
  recomputes detrending and the derived slope channel.
* Attribution-ranked, random-order, and least-important-first controls are
  evaluated on the same samples.
* Outputs are described as kinematic associations, not causal physical forcing.
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

DATE_RE = re.compile(r"^\d{8}$")
LOOKBACK = 24
HORIZON_DAYS = 12
SEED = 42


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv", type=Path, default=Path("data/EGMS_L2b.csv"),
        help="Path to the original EGMS CSV.",
    )
    p.add_argument(
        "--model", type=Path,
        default=Path("outputs/fixed12_training/best_fixed12.keras"),
        help="Final trained fixed-12-day model.",
    )
    p.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/fixed12_training/test_predictions_fixed12.csv"),
        help="Saved predictions for the fixed test sample.",
    )
    p.add_argument(
        "--labels", type=Path,
        default=Path("outputs/fixed12_clustering/cluster_labels_all.npy"),
        help="Unified PCA--KMeans labels for all spatial points.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/fixed12_xai"),
    )
    p.add_argument("--anchor", default=None, help="YYYYMMDD; default first acquisition")
    p.add_argument("--samples-per-group", type=int, default=500)
    p.add_argument("--global-samples", type=int, default=3000)
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument("--ig-batch-size", type=int, default=100)
    p.add_argument("--fidelity-samples", type=int, default=300)
    p.add_argument("--smooth-samples", type=int, default=300)
    p.add_argument("--smooth-draws", type=int, default=8)
    p.add_argument("--smooth-noise-std", type=float, default=0.05)
    p.add_argument("--expected-samples", type=int, default=200_000)
    p.add_argument("--expected-points", type=int, default=594_978)
    p.add_argument("--expected-clusters", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args()


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def fixed12_data(df, anchor_text):
    cols = [str(c) for c in df.columns if DATE_RE.fullmatch(str(c))]
    dates = pd.DatetimeIndex(pd.to_datetime(cols, format="%Y%m%d", errors="raise"))
    order = np.argsort(dates.values); dates = dates[order]; cols = [cols[i] for i in order]
    anchor = pd.to_datetime(anchor_text, format="%Y%m%d") if anchor_text else dates[0]
    if anchor not in dates:
        raise ValueError(f"Anchor {anchor.date()} is not a raw acquisition date")
    count = int((dates[-1] - anchor).days // HORIZON_DAYS) + 1
    grid = pd.date_range(anchor, periods=count, freq=f"{HORIZON_DAYS}D")
    temporal = df[cols].copy(); temporal.columns = dates
    fixed = temporal.reindex(columns=grid)
    observed = np.asarray([d in set(dates) for d in grid], bool)
    values = fixed.ffill(axis=1).fillna(0.0).to_numpy(np.float32)
    return values, grid, observed


def feature(window):
    w = window.astype(np.float32).copy()
    t = np.arange(LOOKBACK, dtype=np.float32)
    A = np.column_stack((t, np.ones_like(t)))
    slope, intercept = np.linalg.lstsq(A, w, rcond=None)[0]
    detrended = w - (slope * t + intercept)
    X = np.stack((detrended, np.diff(detrended, prepend=detrended[0])), axis=-1)
    return np.clip(X, -6.0, 6.0).astype(np.float32)


def feature_batch(raw_windows):
    return np.stack([feature(w) for w in raw_windows], axis=0)


def raw_windows(arr_norm, points, times):
    return np.stack([arr_norm[p, t - LOOKBACK:t] for p, t in zip(points, times)]).astype(np.float32)


def trend_baselines(raw):
    t = np.arange(LOOKBACK, dtype=np.float32)
    tc = t - t.mean()
    centered = raw - raw.mean(axis=1, keepdims=True)
    slopes = np.sum(centered * tc, axis=1, keepdims=True) / np.sum(tc ** 2)
    return (raw.mean(axis=1, keepdims=True) + slopes * tc).astype(np.float32)


def integrated_gradients(model, X, steps):
    X_t = tf.convert_to_tensor(X, tf.float32)
    baseline = tf.zeros_like(X_t)
    gradients = []
    for alpha in tf.linspace(0.0, 1.0, steps + 1):
        interpolated = baseline + alpha * (X_t - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            output = model(interpolated, training=False)
        gradients.append(tape.gradient(output, interpolated))
    # Trapezoidal path integration reduces discretization error.
    stack = tf.stack(gradients, axis=0)
    average = tf.reduce_mean((stack[:-1] + stack[1:]) / 2.0, axis=0)
    return (X_t - baseline) * average


def batched_ig(model, X, steps, batch_size):
    blocks = []
    for start in range(0, len(X), batch_size):
        blocks.append(integrated_gradients(model, X[start:start + batch_size], steps).numpy())
    return np.concatenate(blocks, axis=0)


def gradient_x_input(model, X, batch_size):
    """Compute Grad x Input attributions in batches."""
    blocks = []
    for start in range(0, len(X), batch_size):
        xb = tf.convert_to_tensor(X[start:start + batch_size], tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(xb)
            output = model(xb, training=False)
        grad = tape.gradient(output, xb)
        blocks.append((grad * xb).numpy())
    return np.concatenate(blocks, axis=0)


def choose_rows(pool, n, rng):
    if len(pool) == 0:
        raise ValueError("Cannot sample from an empty XAI group")
    return rng.choice(pool, size=n, replace=len(pool) < n)


def profile_rows(name, group_type, attribution, n_points=None):
    lag_imp = np.mean(np.abs(attribution), axis=(0, 2))
    lag_imp /= lag_imp.sum() + 1e-12
    row = {"Group": name, "Type": group_type, "N_XAI_windows": len(attribution),
           "N_points_full_group": n_points if n_points is not None else np.nan,
           "Recency_top3": float(lag_imp[-3:].sum())}
    for j, value in enumerate(lag_imp):
        row[f"t-{LOOKBACK-j}"] = float(value)
    return row, lag_imp


def coherent_fidelity(model, raw, attribution, y_true_mm, scales_mm,
                      random_order, method_name):
    """Reviewer-ready deletion/insertion with coherent channel rebuilding.

    The normalized trajectory is perturbed first. Detrending and the derived
    slope channel are then recomputed. Forecast degradation is measured as
    MAE in millimetres, while output-change curves are retained as secondary
    diagnostics.
    """
    n = len(raw)
    original_X = feature_batch(raw)
    original_pred_norm = model.predict(
        original_X, batch_size=4096, verbose=0).reshape(-1)
    original_pred_mm = original_pred_norm * scales_mm
    baseline_raw = trend_baselines(raw)
    importance = np.sum(np.abs(attribution), axis=2)
    orders = {
        "attribution_ranked": np.argsort(-importance, axis=1),
        "least_important_first": np.argsort(importance, axis=1),
        "random": random_order,
    }
    curve_rows = []
    for protocol, order in orders.items():
        deletion = raw.copy()
        insertion = baseline_raw.copy()
        for k in range(LOOKBACK + 1):
            pred_del_norm = model.predict(
                feature_batch(deletion), batch_size=4096, verbose=0).reshape(-1)
            pred_ins_norm = model.predict(
                feature_batch(insertion), batch_size=4096, verbose=0).reshape(-1)
            pred_del_mm = pred_del_norm * scales_mm
            pred_ins_mm = pred_ins_norm * scales_mm
            curve_rows.append({
                "Attribution_method": method_name,
                "Protocol": protocol, "Removed_or_inserted_lags": k,
                "Fraction": k / LOOKBACK,
                "Deletion_MAE_mm": float(np.mean(np.abs(pred_del_mm - y_true_mm))),
                "Insertion_MAE_mm": float(np.mean(np.abs(pred_ins_mm - y_true_mm))),
                "Deletion_mean_abs_output_change_mm": float(
                    np.mean(np.abs(pred_del_mm - original_pred_mm))),
                "Insertion_mean_abs_remaining_output_difference_mm": float(
                    np.mean(np.abs(pred_ins_mm - original_pred_mm))),
            })
            if k == LOOKBACK:
                continue
            rows = np.arange(n); lag = order[:, k]
            deletion[rows, lag] = baseline_raw[rows, lag]
            insertion[rows, lag] = raw[rows, lag]
    return pd.DataFrame(curve_rows)


def main():
    args = arguments(); seed_all(args.seed)
    if args.ig_steps < 2:
        raise ValueError("--ig-steps must be at least 2")
    if args.smooth_draws < 1:
        raise ValueError("--smooth-draws must be at least 1")
    csv_path = args.csv.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    pred_path = args.predictions.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    for path in (csv_path, model_path, pred_path, labels_path):
        if not path.is_file(): raise FileNotFoundError(path)

    df = pd.read_csv(csv_path, low_memory=False)
    values, grid, observed = fixed12_data(df, args.anchor)
    T = values.shape[1]; train_end = int(T * 0.70)
    train_values = values[:, :train_end]
    means = np.mean(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    stds = np.std(train_values, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    means[~np.isfinite(means)] = 0; stds[~np.isfinite(stds) | (stds == 0)] = 1
    arr_norm = ((values - means) / stds).astype(np.float32)
    if "VEL" in df.columns:
        activity = np.abs(df["VEL"].to_numpy(float)); activity_source = "absolute VEL column"
    else:
        valid = np.where(observed[1:] & observed[:-1])[0] + 1
        activity = np.median(np.abs(values[:, valid] - values[:, valid - 1]), axis=1)
        activity_source = "median absolute valid 12-day increment"
    del df

    predictions = pd.read_csv(pred_path, low_memory=False)
    required = {"point_index", "target_epoch_index", "target_date",
                "horizon_days", "true_increment_mm"}
    missing = sorted(required - set(predictions.columns))
    if missing: raise ValueError(f"Prediction table lacks: {missing}")
    if len(predictions) != args.expected_samples:
        raise ValueError(
            f"Expected {args.expected_samples:,} prediction rows, "
            f"found {len(predictions):,}"
        )
    if sorted(predictions["horizon_days"].astype(int).unique()) != [12]:
        raise ValueError("XAI prediction rows are not exclusively 12-day-ahead")
    points_all = predictions["point_index"].to_numpy(np.int64)
    times_all = predictions["target_epoch_index"].to_numpy(np.int64)
    if not np.all(observed[times_all] & observed[times_all - 1]):
        raise ValueError("Prediction table contains an imputed or gap-spanning target transition")
    dates_expected = grid[times_all].strftime("%Y-%m-%d")
    if not np.all(predictions["target_date"].astype(str).to_numpy() == dates_expected):
        raise ValueError("Prediction target dates do not match the reconstructed fixed grid")
    raw_true_increment = values[points_all, times_all] - values[points_all, times_all - 1]
    saved_true_increment = predictions["true_increment_mm"].to_numpy(np.float32)
    if not np.allclose(raw_true_increment, saved_true_increment, atol=1e-3, rtol=1e-5):
        raise ValueError("Saved true increments do not match the reconstructed fixed grid")
    labels = np.load(labels_path, allow_pickle=False).astype(int)
    if len(labels) != len(values): raise ValueError("Cluster-label length differs from point count")
    if len(labels) != args.expected_points:
        raise ValueError(
            f"Expected {args.expected_points:,} cluster labels, found {len(labels):,}"
        )
    if points_all.min() < 0 or points_all.max() >= len(values): raise IndexError("Invalid point_index")
    model = tf.keras.models.load_model(model_path)
    rng = np.random.default_rng(args.seed)

    profile_records, plot_profiles = [], []
    clusters = sorted(np.unique(labels).tolist())
    expected_cluster_ids = list(range(args.expected_clusters))
    if clusters != expected_cluster_ids:
        raise ValueError(
            f"Expected cluster IDs {expected_cluster_ids}; found {clusters}"
        )
    for cluster in clusters:
        pool = np.where(labels[points_all] == cluster)[0]
        chosen = choose_rows(pool, args.samples_per_group, rng)
        raw = raw_windows(arr_norm, points_all[chosen], times_all[chosen])
        X = feature_batch(raw)
        ig = batched_ig(model, X, args.ig_steps, args.ig_batch_size)
        row, profile = profile_rows(f"C{cluster}", "Cluster", ig, int(np.sum(labels == cluster)))
        profile_records.append(row); plot_profiles.append((f"C{cluster}", profile))

    q40, q80 = np.quantile(activity, [0.40, 0.80])
    activity_groups = {
        "Stable (<=40th percentile)": activity <= q40,
        "Moderate (40th-80th percentile)": (activity > q40) & (activity < q80),
        "Active (>=80th percentile)": activity >= q80,
    }
    activity_plot = []
    for name, point_mask in activity_groups.items():
        pool = np.where(point_mask[points_all])[0]
        chosen = choose_rows(pool, args.samples_per_group, rng)
        raw = raw_windows(arr_norm, points_all[chosen], times_all[chosen])
        ig = batched_ig(model, feature_batch(raw), args.ig_steps, args.ig_batch_size)
        row, profile = profile_rows(name, "Activity", ig, int(point_mask.sum()))
        profile_records.append(row); activity_plot.append((name, profile))

    chosen_global = choose_rows(np.arange(len(predictions)), args.global_samples, rng)
    raw_global = raw_windows(arr_norm, points_all[chosen_global], times_all[chosen_global])
    X_global = feature_batch(raw_global)
    ig_global = batched_ig(model, X_global, args.ig_steps, args.ig_batch_size)
    gradx_global = gradient_x_input(model, X_global, args.ig_batch_size)
    channel = np.mean(np.abs(ig_global), axis=(0, 1)); channel_pct = 100 * channel / channel.sum()
    channel_lag = np.mean(np.abs(ig_global), axis=0)
    pd.DataFrame({"Lag": [f"t-{LOOKBACK-j}" for j in range(LOOKBACK)],
                  "Detrended_abs_IG": channel_lag[:, 0],
                  "Slope_abs_IG": channel_lag[:, 1]}).to_csv(out_dir / "xai_channel_lag_attribution.csv", index=False)
    pd.DataFrame({"Channel": ["Detrended displacement", "Slope (derived first difference)"],
                  "Total_attribution_pct": channel_pct}).to_csv(out_dir / "xai_channel_summary.csv", index=False)

    smooth_n = min(args.smooth_samples, len(raw_global))
    smooth_total = np.zeros((smooth_n, LOOKBACK, 2), np.float32)
    for _ in range(args.smooth_draws):
        # Noise is added to the trajectory; both engineered channels are rebuilt coherently.
        noisy_raw = raw_global[:smooth_n] + rng.normal(
            0.0, args.smooth_noise_std, size=raw_global[:smooth_n].shape).astype(np.float32)
        smooth_total += batched_ig(
            model, feature_batch(noisy_raw), args.ig_steps, args.ig_batch_size)
    smooth_ig = smooth_total / args.smooth_draws
    ig_lag = np.mean(np.abs(ig_global[:smooth_n]), axis=(0, 2)); ig_lag /= ig_lag.sum() + 1e-12
    smooth_lag = np.mean(np.abs(smooth_ig), axis=(0, 2)); smooth_lag /= smooth_lag.sum() + 1e-12
    pd.DataFrame({"Lag": [f"t-{LOOKBACK-j}" for j in range(LOOKBACK)],
                  "IG_normalized": ig_lag, "SmoothIG_normalized": smooth_lag}).to_csv(
                      out_dir / "xai_global_ig_smoothig_profiles.csv", index=False)

    pred_x = model.predict(X_global, batch_size=4096, verbose=0).reshape(-1)
    pred_0 = model.predict(np.zeros_like(X_global), batch_size=4096, verbose=0).reshape(-1)
    completeness_residual = np.sum(ig_global, axis=(1, 2)) - (pred_x - pred_0)
    pd.DataFrame({"sample_row": chosen_global,
                  "output_difference": pred_x - pred_0,
                  "sum_integrated_gradients": np.sum(ig_global, axis=(1, 2)),
                  "completeness_residual": completeness_residual}).to_csv(
                      out_dir / "ig_completeness.csv", index=False)

    fidelity_n = min(args.fidelity_samples, smooth_n, len(raw_global))
    fidelity_points = points_all[chosen_global[:fidelity_n]]
    fidelity_y_mm = saved_true_increment[chosen_global[:fidelity_n]]
    fidelity_scales_mm = stds[fidelity_points, 0].astype(np.float32)
    random_rng = np.random.default_rng(args.seed + 700)
    shared_random_order = np.stack(
        [random_rng.permutation(LOOKBACK) for _ in range(fidelity_n)])
    fidelity_parts = []
    for method_name, attribution in (
        ("GradXInput", gradx_global[:fidelity_n]),
        ("IntegratedGradients", ig_global[:fidelity_n]),
        ("SmoothIG", smooth_ig[:fidelity_n]),
    ):
        fidelity_parts.append(coherent_fidelity(
            model=model,
            raw=raw_global[:fidelity_n],
            attribution=attribution,
            y_true_mm=fidelity_y_mm,
            scales_mm=fidelity_scales_mm,
            random_order=shared_random_order,
            method_name=method_name,
        ))
    fidelity = pd.concat(fidelity_parts, ignore_index=True)
    fidelity.to_csv(out_dir / "coherent_deletion_insertion_curves.csv", index=False)
    auc_rows = []
    for (method_name, protocol), part in fidelity.groupby(
            ["Attribution_method", "Protocol"]):
        part = part.sort_values("Fraction")
        x = part["Fraction"].to_numpy(float)
        integrate = getattr(np, "trapezoid", np.trapz)
        auc_rows.append({
            "Attribution_method": method_name,
            "Protocol": protocol,
            "Deletion_MAE_AUC_mm": float(integrate(part["Deletion_MAE_mm"], x)),
            "Insertion_MAE_AUC_mm": float(integrate(part["Insertion_MAE_mm"], x)),
            "Deletion_output_change_AUC_mm": float(integrate(
                part["Deletion_mean_abs_output_change_mm"], x)),
            "Insertion_remaining_output_difference_AUC_mm": float(integrate(
                part["Insertion_mean_abs_remaining_output_difference_mm"], x)),
        })
    auc_table = pd.DataFrame(auc_rows)
    auc_table.to_csv(out_dir / "coherent_deletion_insertion_auc.csv", index=False)

    pd.DataFrame(profile_records).to_csv(out_dir / "xai_attribution_profiles.csv", index=False)
    with pd.ExcelWriter(out_dir / "xai_fixed12_summary.xlsx", engine="openpyxl") as writer:
        pd.DataFrame(profile_records).to_excel(writer, "Attribution_Profiles", index=False)
        pd.DataFrame({"Channel": ["Detrended displacement", "Slope"],
                      "Total_attribution_pct": channel_pct}).to_excel(writer, "Channel_Summary", index=False)
        fidelity.to_excel(writer, "Coherent_Fidelity", index=False)
        auc_table.to_excel(writer, "Fidelity_AUC", index=False)

    fig, axes = plt.subplots(1, len(plot_profiles), figsize=(4 * len(plot_profiles), 4), sharey=True)
    if len(plot_profiles) == 1: axes = [axes]
    for ax, (name, profile) in zip(axes, plot_profiles):
        ax.bar(np.arange(LOOKBACK), profile); n_pts = next((r["N_points_full_group"] for r in profile_records if r["Group"]==name), "")
        ax.set_title(f"{name}\n(n={n_pts:,})" if isinstance(n_pts, int) else f"{name}")
        ax.set_xlabel("Lag"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Normalized mean |IG|"); fig.tight_layout()
    fig.savefig(out_dir / "xai_cluster_attribution_fixed12.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (name, profile) in zip(axes, activity_plot):
        ax.bar(np.arange(LOOKBACK), profile); ax.set_title(name); ax.set_xlabel("Lag"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Normalized mean |IG|"); fig.tight_layout()
    fig.savefig(out_dir / "xai_activity_attribution_fixed12.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for c, (ax, title) in enumerate(zip(axes, ["Detrended displacement", "Slope channel"])):
        p = channel_lag[:, c] / (channel_lag[:, c].sum() + 1e-12)
        ax.bar(np.arange(LOOKBACK), p); ax.set_title(f"{title}: {channel_pct[c]:.1f}% total")
        ax.set_xlabel("Lag"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Within-channel normalized |IG|"); fig.tight_layout()
    fig.savefig(out_dir / "xai_channel_attribution_fixed12.png", dpi=200); plt.close(fig)

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(LOOKBACK), ig_lag, marker="o", label="IG")
    plt.plot(np.arange(LOOKBACK), smooth_lag, marker="s", label="Coherent SmoothIG")
    plt.xlabel("Lag"); plt.ylabel("Normalized mean |attribution|")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "xai_global_ig_vs_smoothig.png", dpi=200); plt.close()

    method_order = ["GradXInput", "IntegratedGradients", "SmoothIG"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, method_name in enumerate(method_order):
        method_part = fidelity[fidelity["Attribution_method"] == method_name]
        for protocol, part in method_part.groupby("Protocol"):
            part = part.sort_values("Fraction")
            axes[0, col].plot(part["Fraction"], part["Deletion_MAE_mm"],
                              marker="o", markersize=2, label=protocol)
            axes[1, col].plot(part["Fraction"], part["Insertion_MAE_mm"],
                              marker="o", markersize=2, label=protocol)
        axes[0, col].set_title(f"{method_name}: coherent deletion")
        axes[1, col].set_title(f"{method_name}: coherent insertion")
        for row in range(2):
            axes[row, col].set_xlabel("Fraction of lags")
            axes[row, col].set_ylabel("MAE (mm)")
            axes[row, col].grid(alpha=0.3)
            axes[row, col].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "coherent_deletion_insertion_controls.png", dpi=200)
    plt.close(fig)

    audit = {"model": str(model_path), "predictions": str(pred_path), "labels": str(labels_path),
             "fixed_grid_epochs": len(grid), "xai_source": "saved final fixed-12-day test rows",
             "random_seed": args.seed,
             "ig_steps": args.ig_steps, "ig_baseline": "zero in engineered feature space",
             "ig_path_coherence": "scaling both channels from zero preserves the derived slope relation",
             "mean_absolute_completeness_residual": float(np.mean(np.abs(completeness_residual))),
             "smoothig_samples": smooth_n, "smoothig_draws": args.smooth_draws,
             "smoothig_noise_std_normalized_trajectory": args.smooth_noise_std,
             "smoothig_coherence": "noise applied to trajectory before rebuilding detrended and slope channels",
             "activity_source": activity_source, "activity_q40": float(q40), "activity_q80": float(q80),
             "fidelity_samples": fidelity_n,
             "coherent_perturbation": "modify raw normalized trajectory, then recompute detrending and slope",
             "fidelity_primary_metric": "MAE in millimetres against saved true increments",
             "fidelity_attribution_methods": ["GradXInput", "IntegratedGradients", "SmoothIG"],
             "controls": ["attribution-ranked", "random", "least-important-first"],
             "interpretation_boundary": "kinematic association only; no hydrological or causal inference",
             "metric_space": "attribution computed on increment-space predictions (delta mode); R2_delta is the primary performance metric for this model"}
    (out_dir / "xai_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[DONE] Reviewer-ready coherent XAI outputs written to {out_dir}")


if __name__ == "__main__":
    main()

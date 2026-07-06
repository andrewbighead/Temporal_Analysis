# -*- coding: utf-8 -*-
"""
xai_enhanced.py
================
Explainability analyses for:
  "Machine Learning and Explainable Deep Learning for Forecasting
   Landslide Deformation using EGMS Time Series"
  Madadi et al., Applied Soft Computing (2025)

Description:
  Computes Integrated Gradients (IG) attribution profiles stratified by:
    1. Deformation cluster (PCA-KMeans, 4 clusters) — Figure 14
    2. Activity level (active / moderate / stable)   — Figure 15
    3. Input channel (detrended vs slope)             — Figure 16

Prerequisites (must be in the same folder as this script):
  - best_lstm.keras          : trained CNN-BiLSTM-Attention model
  - cluster_labels_all.npy   : unified cluster assignment (from fix_cluster_assignment.py)
  - EGMS CSV data file       : set CSV_PATH below

Outputs:
  - xai_cluster_attribution.png   : Figure 14 (1x4 layout)
  - xai_activity_attribution.png  : Figure 15
  - xai_channel_attribution.png   : Figure 16
  - xai_enhanced_summary.xlsx     : full attribution numbers

Usage:
  python xai_enhanced.py

Note on reproducibility:
  Cluster labels are loaded from cluster_labels_all.npy (produced by
  fix_cluster_assignment.py) to ensure consistency with Table 10 and
  Figure 14 in the paper. Do NOT recompute KMeans here.
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# =============================================================================
# CONFIG — edit these paths
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = SCRIPT_DIR
CSV_PATH     = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
MODEL_PATH   = os.path.join(SCRIPT_DIR, "best_lstm.keras")
LABELS_PATH  = os.path.join(SCRIPT_DIR, "cluster_labels_all.npy")

# Must match training config exactly
LOOKBACK          = 24
CHANNELS          = 2
DETREND_WINDOWS   = True
USE_SLOPE_CHANNEL = True
WINSOR_X          = 6.0
BIAS_POWER        = 2.0
KMEANS_CLUSTERS   = 4
N_PER_GROUP       = 500    # windows per group for attribution
XAI_SAMPLES       = 3000   # windows for global channel attribution
IG_STEPS          = 32
SEED              = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

def figpath(name): return os.path.join(OUT_DIR, name)
def savefig(p):
    plt.tight_layout()
    plt.savefig(p, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {os.path.basename(p)}")

# =============================================================================
# 1. Load data
# =============================================================================
print("[1/6] Loading data...")
df = pd.read_csv(CSV_PATH)
date_cols = sorted([c for c in df.columns if re.fullmatch(r"\d{8}", str(c))])
ts_raw  = df[date_cols].copy()
ts_ff   = ts_raw.ffill(axis=1).fillna(0.0).astype(np.float32)
arr_vals = ts_ff.to_numpy(np.float32)
n_points, T = arr_vals.shape

# Velocity for activity stratification
vel_col = "VEL" if "VEL" in df.columns else None
vel = df[vel_col].to_numpy(np.float32) if vel_col else \
      np.gradient(arr_vals, axis=1).mean(axis=1).astype(np.float32)

# Train-only normalization (matches training pipeline exactly)
train_end = int(T * 0.70)
val_end   = int(T * 0.85)
train_obs = arr_vals[:, :train_end]
means = np.nanmean(train_obs, axis=1, keepdims=True)
stds  = np.nanstd(train_obs,  axis=1, keepdims=True)
stds[stds == 0] = 1.0
ts_norm = ((arr_vals - means) / stds).astype(np.float32)
print(f"   n_points={n_points:,}, T={T}, test_range=[{val_end},{T})")

# =============================================================================
# 2. Load model
# =============================================================================
print("[2/6] Loading model...")
assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
model = models.load_model(MODEL_PATH)
print(f"   Loaded: {MODEL_PATH}")

# =============================================================================
# 3. Feature engineering (must match training exactly)
# =============================================================================
def build_window_channels(w_norm):
    w = w_norm.astype(np.float32).copy()
    if DETREND_WINDOWS:
        t_idx = np.arange(LOOKBACK, dtype=np.float32)
        A = np.vstack([t_idx, np.ones_like(t_idx)]).T
        a, b = np.linalg.lstsq(A, w, rcond=None)[0]
        w = w - (a * t_idx + b)
    outs = [w]
    if USE_SLOPE_CHANNEL:
        outs.append(np.diff(w, prepend=w[0]))
    Xw = np.stack(outs, axis=-1)
    if WINSOR_X:
        Xw = np.clip(Xw, -WINSOR_X, WINSOR_X)
    return Xw

def sample_windows(pidx_pool, n, seed_offset=0):
    """Sample n windows from a pool of point indices."""
    rng = np.random.default_rng(SEED + seed_offset)
    t_low, t_high = val_end, T - 1
    sel = pidx_pool[rng.integers(0, len(pidx_pool), size=n)]
    u   = rng.random(n) ** BIAS_POWER
    tidx = (t_low + (t_high - t_low - 1) * u).astype(int)
    X = np.zeros((n, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros(n, np.float32)
    for i, (pi, ti) in enumerate(zip(sel, tidx)):
        X[i] = build_window_channels(ts_norm[pi, ti - LOOKBACK:ti])
        y[i] = ts_norm[pi, ti] - ts_norm[pi, ti - 1]
    return X, y, sel, tidx

# =============================================================================
# 4. Attribution functions
# =============================================================================
def integrated_gradients(x_np, steps=IG_STEPS, baseline=0.0):
    x_t    = tf.constant(x_np, dtype=tf.float32)
    base_t = tf.constant(np.full_like(x_np, baseline), dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, steps + 1)
    grads  = []
    for a in alphas:
        interp = base_t + a * (x_t - base_t)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = model(interp, training=False)
        grads.append(tape.gradient(pred, interp).numpy())
    avg_grads = np.mean(grads, axis=0)
    return (x_np - baseline) * avg_grads   # (B, L, C)

def mean_lag_profile(X_np, batch=100):
    """Compute normalized mean |IG| per lag, collapsed over channels."""
    all_ig = []
    for i in range(0, len(X_np), batch):
        ig = integrated_gradients(X_np[i:i+batch])
        all_ig.append(np.abs(ig).mean(axis=-1))   # (b, L)
    profile = np.concatenate(all_ig, axis=0).mean(axis=0)
    return profile / (profile.sum() + 1e-10)

# =============================================================================
# 5. Load unified cluster labels
# =============================================================================
print("[3/6] Loading cluster labels...")
assert os.path.exists(LABELS_PATH), \
    f"cluster_labels_all.npy not found: {LABELS_PATH}\n" \
    f"Run fix_cluster_assignment.py first."
all_labels = np.load(LABELS_PATH).astype(int)
assert len(all_labels) == n_points
print("   Cluster counts:")
for c in range(KMEANS_CLUSTERS):
    print(f"     C{c}: {np.sum(all_labels==c):,}")

# =============================================================================
# 6. Figure 14 — Attribution stratified by cluster (1×4 layout)
# =============================================================================
print("[4/6] Computing cluster-stratified attribution (Figure 14)...")

cluster_names = {
    0: "Quasi-stable (C0)",
    1: "Oscillatory (C1)",
    2: "Progressive (C2)",
    3: "Transitional (C3)",
}

fig, axes = plt.subplots(1, KMEANS_CLUSTERS, figsize=(16, 4), sharey=True)
cluster_profiles = {}

for c in range(KMEANS_CLUSTERS):
    pool = np.where(all_labels == c)[0]
    X_c, _, _, _ = sample_windows(pool, N_PER_GROUP, seed_offset=c)
    profile = mean_lag_profile(X_c)
    cluster_profiles[c] = profile

    ax = axes[c]
    ax.bar(range(LOOKBACK), profile, color=f"C{c}")
    ax.set_title(f"{cluster_names[c]}\n(n={len(pool):,})", fontsize=9)
    ax.set_xticks(range(0, LOOKBACK, 4))
    ax.set_xticklabels([f"t-{LOOKBACK-j}" for j in range(0, LOOKBACK, 4)],
                       fontsize=7, rotation=45)
    ax.set_xlabel("Lag")
    if c == 0:
        ax.set_ylabel("Norm. |IG|")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Temporal lag attribution profiles by deformation cluster", fontsize=11)
savefig(figpath("xai_cluster_attribution.png"))

# =============================================================================
# 7. Figure 15 — Attribution by activity level
# =============================================================================
print("[5/6] Computing activity-stratified attribution (Figure 15)...")

abs_vel = np.abs(vel)
q80 = np.percentile(abs_vel, 80)
q40 = np.percentile(abs_vel, 40)

activity_groups = {
    f"Active\n(|vel| ≥ 80th pct\nn={np.sum(abs_vel>=q80):,})":
        np.where(abs_vel >= q80)[0],
    f"Moderate\n(40th–80th pct\nn={np.sum((abs_vel>q40)&(abs_vel<q80)):,})":
        np.where((abs_vel > q40) & (abs_vel < q80))[0],
    f"Stable\n(|vel| ≤ 40th pct\nn={np.sum(abs_vel<=q40):,})":
        np.where(abs_vel <= q40)[0],
}
act_colors = ["#F44336", "#FF9800", "#2196F3"]
act_profiles = {}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, (label, pool), color in zip(axes, activity_groups.items(), act_colors):
    X_g, _, _, _ = sample_windows(pool, N_PER_GROUP, seed_offset=10)
    profile = mean_lag_profile(X_g)
    act_profiles[label] = profile
    ax.bar(range(LOOKBACK), profile, color=color, alpha=0.85)
    ax.set_title(label, fontsize=9)
    ax.set_xticks(range(0, LOOKBACK, 4))
    ax.set_xticklabels([f"t-{LOOKBACK-j}" for j in range(0, LOOKBACK, 4)],
                       fontsize=7, rotation=45)
    ax.set_xlabel("Lag")
    if ax is axes[0]:
        ax.set_ylabel("Norm. |IG|")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Temporal lag attribution profiles by activity level", fontsize=11)
savefig(figpath("xai_activity_attribution.png"))

# =============================================================================
# 8. Figure 16 — Channel-level attribution
# =============================================================================
print("[5b/6] Computing channel-level attribution (Figure 16)...")

rng_g = np.random.default_rng(SEED + 99)
pidx_g = rng_g.integers(0, n_points, XAI_SAMPLES)
t_low_g, t_high_g = val_end, T - 1
u_g = rng_g.random(XAI_SAMPLES) ** BIAS_POWER
tidx_g = (t_low_g + (t_high_g - t_low_g - 1) * u_g).astype(int)
X_global = np.zeros((XAI_SAMPLES, LOOKBACK, CHANNELS), np.float32)
for i, (pi, ti) in enumerate(zip(pidx_g, tidx_g)):
    X_global[i] = build_window_channels(ts_norm[pi, ti - LOOKBACK:ti])

ig_full = np.concatenate([
    integrated_gradients(X_global[i:i+100])
    for i in range(0, XAI_SAMPLES, 100)
], axis=0)   # (N, L, C)

ch0_imp = np.mean(np.abs(ig_full[:, :, 0]), axis=0)
ch1_imp = np.mean(np.abs(ig_full[:, :, 1]), axis=0)
total   = ch0_imp.sum() + ch1_imp.sum()
ch0_pct = 100 * ch0_imp.sum() / total
ch1_pct = 100 * ch1_imp.sum() / total

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, imp, color, title in zip(
    axes,
    [ch0_imp / ch0_imp.sum(), ch1_imp / ch1_imp.sum()],
    ["#1565C0", "#E65100"],
    ["Channel 1: Detrended Displacement",
     "Channel 2: Slope (First-order Differences)"]
):
    ax.bar(range(LOOKBACK), imp, color=color, alpha=0.85)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(0, LOOKBACK, 4))
    ax.set_xticklabels([f"t-{LOOKBACK-j}" for j in range(0, LOOKBACK, 4)],
                       fontsize=7, rotation=45)
    ax.set_ylabel("Norm. |IG|")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(f"Channel-level attribution  |  "
             f"Detrended: {ch0_pct:.1f}%   Slope: {ch1_pct:.1f}%",
             fontsize=11)
savefig(figpath("xai_channel_attribution.png"))

# =============================================================================
# 9. Save summary Excel
# =============================================================================
print("[6/6] Saving summary Excel...")

lag_names = [f"t-{LOOKBACK-j}" for j in range(LOOKBACK)]
rows = []

for c, profile in cluster_profiles.items():
    row = {"Group": cluster_names[c], "Type": "Cluster",
           "N_points": int(np.sum(all_labels == c))}
    for j, lag in enumerate(lag_names):
        row[lag] = round(float(profile[j]), 6)
    row["Recency_top3"] = round(float(profile[-3:].sum()), 4)
    rows.append(row)

for label, profile in act_profiles.items():
    row = {"Group": label.replace("\n", " "), "Type": "Activity"}
    for j, lag in enumerate(lag_names):
        row[lag] = round(float(profile[j]), 6)
    row["Recency_top3"] = round(float(profile[-3:].sum()), 4)
    rows.append(row)

for ch_name, imp in [("Detrended", ch0_imp), ("Slope", ch1_imp)]:
    p = imp / imp.sum()
    row = {"Group": ch_name, "Type": "Channel"}
    for j, lag in enumerate(lag_names):
        row[lag] = round(float(p[j]), 6)
    row["Recency_top3"] = round(float(p[-3:].sum()), 4)
    rows.append(row)

df_out = pd.DataFrame(rows)
xl_path = figpath("xai_enhanced_summary.xlsx")
with pd.ExcelWriter(xl_path, engine="openpyxl") as writer:
    df_out.to_excel(writer, sheet_name="Attribution_Profiles", index=False)
    pd.DataFrame({
        "Channel": ["Detrended Displacement", "Slope (First-order diff)"],
        "Total_Attribution_%": [round(ch0_pct, 2), round(ch1_pct, 2)]
    }).to_excel(writer, sheet_name="Channel_Summary", index=False)
print(f"   Saved: xai_enhanced_summary.xlsx")

# =============================================================================
# 10. Print summary
# =============================================================================
print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print("\nCluster recency (last 3 lags):")
for c, profile in cluster_profiles.items():
    r = profile[-3:].sum()
    print(f"  {cluster_names[c]}: {r:.3f} ({r*100:.1f}%)")

print("\nActivity level recency (last 3 lags):")
for label, profile in act_profiles.items():
    r = profile[-3:].sum()
    print(f"  {label.replace(chr(10),' ')}: {r:.3f} ({r*100:.1f}%)")

print(f"\nChannel attribution:")
print(f"  Detrended: {ch0_pct:.1f}%")
print(f"  Slope:     {ch1_pct:.1f}%")
print("=" * 55)
print("\nDone.")

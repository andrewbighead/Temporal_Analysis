# -*- coding: utf-8 -*-
"""
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Landslide Deformation using EGMS Time Series"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing 

recompute_cluster_stratification_v2.py
========================================
FIXED VERSION: Uses the SAME fixed 200k sample as the main evaluation,
then FILTERS by cluster label to get per-cluster metrics.

This matches the original approach:
  n_test[c] = number of windows in the 200k sample whose spatial point
               belongs to cluster c (NOT a separate sample per cluster).
"""

import os, re, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
tf.random.set_seed(SEED)   # set after TF import for full reproducibility
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================== EDIT THESE PATHS IF NEEDED ======================
CSV_PATH    = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
MODEL_PATH  = r"E:\خروجی های پروژه\outputs_LSTM_BiLSTM\outputs\best_lstm.keras"
LABELS_PATH = r"E:\خروجی های پروژه\outputs_LSTM_BiLSTM\outputs\cluster_labels_all.npy"
OUT_DIR     = os.path.dirname(MODEL_PATH)
# ========================================================================

LOOKBACK   = 24
CHANNELS   = 2
DETREND    = True
SLOPE_CH   = True
WINSOR_X   = 6.0
BIAS_POWER = 2.0
SEED       = 42
N_TEST     = 200_000
KMEANS_K   = 4

def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

print("[INFO] Loading CSV ...")
df = pd.read_csv(CSV_PATH)
date_cols = sorted([c for c in df.columns if _is_date_col(c)])
ts = df[date_cols].copy()
ts_ff = ts.ffill(axis=1)
ts_clean = ts_ff.fillna(0.0).astype(np.float32)
arr_vals = ts_clean.to_numpy(np.float32)
n_points, T = arr_vals.shape
train_end = int(T * 0.70)
val_end   = int(T * 0.85)
print(f"[INFO] n_points={n_points:,}, T={T}")

train_obs = ts_ff.iloc[:, :train_end].to_numpy(np.float32)
means = np.nanmean(train_obs, axis=1, keepdims=True)
stds  = np.nanstd(train_obs,  axis=1, keepdims=True)
means = np.nan_to_num(means, nan=0.0)
stds  = np.nan_to_num(stds,  nan=1.0)
stds[stds == 0] = 1.0
ts_norm = ((arr_vals - means) / stds).astype(np.float32)
TE_RANGE = (val_end, T)

# Load unified cluster labels
print(f"[INFO] Loading cluster labels ...")
all_labels = np.load(LABELS_PATH)
assert len(all_labels) == n_points

def build_window(w_norm):
    w = w_norm.copy()
    if DETREND:
        t = np.arange(LOOKBACK, dtype=np.float32)
        A = np.vstack([t, np.ones_like(t)]).T
        a, b = np.linalg.lstsq(A, w, rcond=None)[0]
        w = w - (a*t + b)
    outs = [w]
    if SLOPE_CH:
        outs.append(np.diff(w, prepend=w[0]))
    Xw = np.stack(outs, axis=-1)
    if WINSOR_X:
        Xw = np.clip(Xw, -WINSOR_X, WINSOR_X)
    return Xw

# ── Build the SAME fixed 200k sample as main evaluation ──────────────────
print(f"[INFO] Building fixed 200k sample (seed={SEED}, p={BIAS_POWER}) ...")
rng = np.random.default_rng(SEED)
t_low, t_high = TE_RANGE
u = rng.random(N_TEST) ** BIAS_POWER
tidx = (t_low + (t_high - t_low - 1)*u).astype(int)
pidx = rng.integers(0, n_points, size=N_TEST)

X_all = np.zeros((N_TEST, LOOKBACK, CHANNELS), np.float32)
y_all = np.zeros(N_TEST, np.float32)
for i, (pi, ti) in enumerate(zip(pidx, tidx)):
    X_all[i] = build_window(ts_norm[pi, ti-LOOKBACK:ti])
    y_all[i] = ts_norm[pi, ti] - ts_norm[pi, ti-1]

print("[INFO] Loading model and running inference ...")
model = tf.keras.models.load_model(MODEL_PATH)
pred_all = model.predict(X_all, verbose=0, batch_size=4096).ravel()

# Denormalize
stds_s  = stds[pidx].reshape(-1)
means_s = means[pidx].reshape(-1)
last_den = ts_norm[pidx, tidx-1] * stds_s + means_s
true_level = last_den + y_all * stds_s
pred_level = last_den + pred_all * stds_s

# Cluster label for each window (from the spatial point)
window_cluster = all_labels[pidx]

# ── TABLE 10: filter by cluster ───────────────────────────────────────────
print("\n[1/2] Computing Table 10 (per-cluster metrics from fixed 200k sample) ...")
table10_rows = []
cluster_labels_tex = {0:"quasi-stable", 1:"oscillatory", 2:"progressive", 3:"transitional"}

for c in range(KMEANS_K):
    mask = window_cluster == c
    n_test_c = int(mask.sum())
    n_pts_c  = int(np.sum(all_labels == c))

    if n_test_c == 0:
        print(f"  C{c}: no windows — skipping")
        continue

    mae  = mean_absolute_error(true_level[mask], pred_level[mask])
    rmse = math.sqrt(mean_squared_error(true_level[mask], pred_level[mask]))
    r2   = r2_score(true_level[mask], pred_level[mask])

    print(f"  C{c} ({cluster_labels_tex[c]}): n_pts={n_pts_c:,}  n_test={n_test_c:,}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    table10_rows.append({
        "Cluster": c, "Label": cluster_labels_tex[c],
        "n_points": n_pts_c, "n_test": n_test_c,
        "MAE": round(mae,3), "RMSE": round(rmse,3), "R2": round(r2,3)
    })

df_t10 = pd.DataFrame(table10_rows)
out_t10 = os.path.join(OUT_DIR, "table10_cluster_performance_v2.csv")
df_t10.to_csv(out_t10, index=False)
print(f"[SAVED] {out_t10}")

# Print LaTeX
print("\n" + "="*65)
print("LaTeX for Table 10 (copy to Overleaf):")
print("="*65)
for row in table10_rows:
    c = row["Cluster"]
    print(f"  {c} ({row['Label']:<12}) & {row['n_points']:,} & {row['n_test']:,} & {row['MAE']} & {row['RMSE']} & {row['R2']} \\\\")

# ── FIGURE 14: XAI per-cluster attribution ───────────────────────────────
print("\n[2/2] Computing Figure 14 (XAI cluster attribution) ...")

def integrated_gradients(model, x, steps=64, baseline=0.0):
    x_t = tf.constant(x, dtype=tf.float32)
    base_t = tf.constant(np.full_like(x, baseline), dtype=tf.float32)
    alphas = tf.linspace(0.0, 1.0, steps+1)
    grads = []
    for a in alphas:
        interp = base_t + a*(x_t - base_t)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = model(interp, training=False)
        g = tape.gradient(pred, interp)
        grads.append(g.numpy())
    avg_g = np.mean(grads, axis=0)
    return (x - baseline) * avg_g

N_PER_GROUP = 500
cluster_names = {0:"Quasi-stable (C0)", 1:"Oscillatory (C1)",
                 2:"Progressive (C2)", 3:"Transitional (C3)"}

fig, axes = plt.subplots(1, KMEANS_K, figsize=(16,4), sharey=True)
for c in range(KMEANS_K):
    pool = np.where(all_labels == c)[0]
    rng2 = np.random.default_rng(SEED + c + 100)
    sel = pool[rng2.integers(0, len(pool), size=N_PER_GROUP)]
    u2 = rng2.random(N_PER_GROUP) ** BIAS_POWER
    tidx2 = (TE_RANGE[0] + (TE_RANGE[1]-TE_RANGE[0]-1)*u2).astype(int)

    X_c = np.zeros((N_PER_GROUP, LOOKBACK, CHANNELS), np.float32)
    for i, (pi, ti) in enumerate(zip(sel, tidx2)):
        X_c[i] = build_window(ts_norm[pi, ti-LOOKBACK:ti])

    igs = []
    for i in range(0, N_PER_GROUP, 50):
        ig = integrated_gradients(model, X_c[i:i+50])
        igs.append(np.abs(ig).mean(axis=-1))
    profile = np.concatenate(igs).mean(axis=0)
    profile /= (profile.sum() + 1e-10)

    ax = axes[c]
    ax.bar(range(LOOKBACK), profile, color=f"C{c}")
    ax.set_title(f"{cluster_names[c]}\n(n={len(pool):,})", fontsize=9)
    ax.set_xticks(range(0, LOOKBACK, 4))
    ax.set_xticklabels([f"t-{LOOKBACK-j}" for j in range(0, LOOKBACK, 4)],
                       fontsize=7, rotation=45)
    ax.set_xlabel("Lag")
    if c == 0: ax.set_ylabel("Norm. |IG|")
    ax.grid(axis='y', alpha=0.3)

fig.suptitle("Temporal lag attribution profiles by deformation cluster", fontsize=11)
plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "xai_cluster_attribution.png")
plt.savefig(out_fig, dpi=180, bbox_inches='tight')
plt.close()
print(f"[SAVED] {out_fig}")

print("\n" + "="*65)
print("DONE. Use table10_cluster_performance_v2.csv for Table 10.")
print("="*65)

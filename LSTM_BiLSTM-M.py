# -*- coding: utf-8 -*-
"""
LSTM_BiLSTM-M.py — Main training script
=========================================
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Landslide Deformation using EGMS Time Series"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing 

Complete end-to-end pipeline:
  - Data loading and chronological train/val/test split (70/15/15)
  - Per-point normalization using train-only statistics (no data leakage)
  - Feature engineering: local detrending + slope-encoded auxiliary channel
  - Recency-biased temporal sampling with non-uniform window selection
  - Models (set MODEL_VARIANT): "lstm", "bilstm", "cnn_bilstm_attn" (default),
    "tcn_bilstm_attn"
  - Training: AdamW + CosineDecayRestarts + Huber loss + gradient clipping
  - Evaluation: MAE/RMSE/R² on fixed 200,000-window test sample (denormalized)
  - Visualization suite: spatial maps, cluster trends, error diagnostics,
    per-point MAE, Folium interactive maps, GIF animation
  - XAI: Grad×Input, Integrated Gradients, SmoothIG, permutation importance,
    deletion/insertion AUC curves, spatial recent-lag reliance map

IMPORTANT — Cluster labels for Table 10 and Figure 14:
  This script runs KMeans internally for visualization only.
  For the per-cluster forecasting metrics (Table 10) and XAI attribution
  profiles (Figure 14) reported in the paper, use the unified cluster
  assignment produced by fix_cluster_assignment.py (cluster_labels_all.npy)
  and recompute_cluster_stratification_v2.py respectively. This ensures
  consistency between Table 10 and Figure 14.

Usage:
  Set CSV_PATH and OUT_DIR below, then run:
    python LSTM_BiLSTM-M.py

Tested headless (no plt.show). Compatible with Windows and Linux.
"""

import os, sys, re, math, time, random, warnings, subprocess
from typing import Tuple
import numpy as np
import pandas as pd

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# quiet TF & offline folium assets
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["FOLIUM_DISABLE_CDN"] = "1"  # bundle assets in HTML

trapezoid = getattr(np, "trapezoid", np.trapz)  # np.trapz deprecated in new numpy


# ====================== CONFIG ======================
CSV_PATH    = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
OUT_DIR     = None  # None -> alongside script in "outputs"

# context / target
LOOKBACK    = 24     # try 36 for higher accuracy (memory permitting)
PRED_STEPS  = 1
TARGET_MODE = "delta"    # "delta" (recommended for cumulative) or "level"

# feature engineering
DETREND_WINDOWS    = True
USE_SLOPE_CHANNEL  = True   # add second channel: first differences of detrended series
WINSOR_X           = 6.0    # winsorize normalized inputs to [-6,6] to tame outliers
LABEL_CLIP_SIGMA   = 3.0    # clip |y| <= 3*std within each batch (robust to rare spikes)

# model / training
MODEL_VARIANT      = "cnn_bilstm_attn"  # choices: "lstm", "bilstm", "cnn_bilstm_attn", "tcn_bilstm_attn"
EPOCHS             = 60
BATCH_SIZE         = 1024
STEPS_PER_EPOCH    = 600
VAL_STEPS          = 120

# sampling
RECENCY_BIASED     = True   # bias times toward recent (matches eval distribution)
BIAS_POWER         = 2.0    # 2.0 => quadratic bias to recent timesteps

# viz & XAI
XAI_SAMPLES        = 3000
PERM_TRIALS        = 3
RECENT_K           = 3
MAX_CORR_STEPS     = 120
MISSING_SAMPLE_N   = 1000
TOP_VEL_K          = 5
RUN_KMEANS         = True
KMEANS_CLUSTERS    = 4
FOLIUM_SAMPLE      = 80000   # max points for marker clustering HTML (browser-friendly)
ENABLE_SHAP        = False   # SHAP is heavy; enable if you want a small beeswarm

SEED               = 42

# ====================== UTILITIES ======================
def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
seed_all()

def script_dir():
    return os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

OUT_DIR = OUT_DIR or os.path.join(script_dir(), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"[INFO] Outputs -> {OUT_DIR}")

def savefig(path, dpi=180): 
    plt.savefig(path, dpi=dpi, bbox_inches="tight"); 
    plt.close()
def figpath(name): 
    return os.path.join(OUT_DIR, name)
def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

def _pip_install(pkgs):
    import subprocess, sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)
    except Exception as e: 
        print("[WARN] pip install failed:", pkgs, "->", e)

for pk in ["tensorflow","scikit-learn","folium","branca"]:
    try: 
        __import__(pk.split("==")[0])
    except Exception: 
        _pip_install([pk])

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, models
tf.random.set_seed(SEED)   # set after TF import for full reproducibility
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import folium
from folium import plugins
from branca.colormap import linear, LinearColormap


# ====================== LOAD DATA ======================
assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Using CSV: {CSV_PATH}")

for col in ("latitude","longitude"):
    if col not in df.columns:
        raise ValueError(f"CSV must contain '{col}' column.")

date_cols = sorted([c for c in df.columns if _is_date_col(c)])
assert len(date_cols)>0, "No date-like columns (YYYYMMDD) found."

ts = df[date_cols].copy()
ts_orig = ts.copy()

# (Option 2) Causal imputation: forward-fill only
ts_ff = ts.ffill(axis=1)  # internal gaps filled causally; leading gaps may remain NaN

# Keep a fully-defined series for downstream windowing (avoid NaNs in arr_vals)
ts_clean = ts_ff.fillna(0.0).astype(np.float32)
arr_vals = ts_clean.to_numpy(np.float32)

n_points, T = ts_clean.shape
train_end = int(T*0.70); val_end = int(T*0.85)
print(f"[INFO] n_points={n_points:,}, timesteps={T}, lookback={LOOKBACK}")
print(f"[INFO] Target windows -> train[{LOOKBACK},{train_end}), val[{train_end},{val_end}), test[{val_end},{T})")

# Per-point stats on TRAIN only, using observed values only (ignore NaNs)
train_vals_obs = ts_ff.iloc[:, :train_end].to_numpy(np.float32)  # may contain NaNs for leading gaps
means = np.nanmean(train_vals_obs, axis=1, keepdims=True)
stds  = np.nanstd(train_vals_obs,  axis=1, keepdims=True)

# Safety for points with no observed values in TRAIN
means = np.nan_to_num(means, nan=0.0)
stds  = np.nan_to_num(stds,  nan=1.0)
stds[stds == 0] = 1.0

to_norm   = lambda a: (a - means) / stds
to_denorm = lambda a: a * stds + means

ts_norm = to_norm(arr_vals).astype(np.float32)

date_index = pd.to_datetime(date_cols, format="%Y%m%d")
lons, lats = df["longitude"].values, df["latitude"].values

# channels based on config
CHANNELS = 1 + int(USE_SLOPE_CHANNEL)

# ====================== WINDOW HELPERS ======================
def build_window_channels(w_norm: np.ndarray) -> np.ndarray:
    """
    Given a normalized window (len=LOOKBACK), return (LOOKBACK, CHANNELS)
    channel 0: detrended (if DETREND_WINDOWS) else original normalized window
    channel 1: (optional) first differences of channel 0
    """
    w = w_norm.astype(np.float32).copy()
    if DETREND_WINDOWS:
        t_idx = np.arange(LOOKBACK, dtype=np.float32)
        A = np.vstack([t_idx, np.ones_like(t_idx)]).T
        a, b = np.linalg.lstsq(A, w, rcond=None)[0]
        w0 = w - (a * t_idx + b)
    else:
        w0 = w
    outs = [w0]
    if USE_SLOPE_CHANNEL:
        slope = np.diff(w0, prepend=w0[0])
        outs.append(slope)
    Xw = np.stack(outs, axis=-1)  # (L,C)
    if WINSOR_X is not None:
        Xw = np.clip(Xw, -float(WINSOR_X), float(WINSOR_X))
    return Xw

# ====================== DATASETS (repeatable) ======================
TR_RANGE = (LOOKBACK, train_end - (PRED_STEPS-1))
VA_RANGE = (train_end, val_end - (PRED_STEPS-1))
TE_RANGE = (val_end,  T - (PRED_STEPS-1))

def sample_times(t_low, t_high, size):
    if not RECENCY_BIASED:
        return np.random.randint(t_low, t_high, size=size)
    # quadratic bias toward recent
    u = np.random.rand(size) ** BIAS_POWER
    return (t_low + (t_high - t_low - 1) * u).astype(int)

def sample_batch(arr_norm, time_range, batch_size, mode="delta"):
    nP, TT = arr_norm.shape
    t_low, t_high = time_range
    t = sample_times(t_low, t_high, batch_size)
    p = np.random.randint(0, nP, size=(batch_size,))
    X = np.zeros((batch_size, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((batch_size, PRED_STEPS), np.float32)
    for i,(pi,ti) in enumerate(zip(p,t)):
        w = arr_norm[pi, ti-LOOKBACK:ti]
        X[i,:,:] = build_window_channels(w)
        if mode=="delta": y[i,0] = arr_norm[pi, ti] - arr_norm[pi, ti-1]
        else:             y[i,0] = arr_norm[pi, ti]
    # robust label clipping
    if LABEL_CLIP_SIGMA is not None:
        y_std = float(np.std(y)+1e-8)
        clip = max(0.5, LABEL_CLIP_SIGMA * y_std)
        y = np.clip(y, -clip, clip)
    return X,y

def make_dataset(arr_norm, time_range, batch_size, mode="delta"):
    output_signature = (
        tf.TensorSpec(shape=(None, LOOKBACK, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(None, PRED_STEPS), dtype=tf.float32),
    )
    def gen():
        while True:
            yield sample_batch(arr_norm, time_range, batch_size, mode)
    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return ds.repeat().prefetch(2)

train_ds = make_dataset(ts_norm, TR_RANGE, BATCH_SIZE, TARGET_MODE)
val_ds   = make_dataset(ts_norm, VA_RANGE, BATCH_SIZE, TARGET_MODE)

# fixed test sample (metrics & XAI)
def make_fixed_test(arr_norm, time_range, n, mode="delta", seed=SEED):
    rng = np.random.default_rng(seed)
    t_low, t_high = time_range
    t = sample_times(t_low, t_high, n)
    p = rng.integers(0, arr_norm.shape[0], size=n)
    X = np.zeros((n, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((n, PRED_STEPS), np.float32)
    for i,(pi,ti) in enumerate(zip(p,t)):
        w = arr_norm[pi, ti-LOOKBACK:ti]
        X[i,:,:] = build_window_channels(w)
        y[i,0] = (arr_norm[pi, ti]-arr_norm[pi, ti-1]) if mode=="delta" else arr_norm[pi, ti]
    return X,y,p,t

X_test, y_test, pidx_test, tidx_test = make_fixed_test(ts_norm, TE_RANGE, n=max(200_000, XAI_SAMPLES), mode=TARGET_MODE)

# ====================== MODELS ======================
def build_model(variant="cnn_bilstm_attn", lookback=LOOKBACK, channels=CHANNELS):
    inp = layers.Input(shape=(lookback, channels))

    if variant == "lstm":
        x = layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(inp)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation="gelu")(x)
        x = layers.Dropout(0.25)(x)

    elif variant == "bilstm":
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(inp)
        x = layers.LayerNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(96, activation="gelu")(x)
        x = layers.Dropout(0.25)(x)

    elif variant == "cnn_bilstm_attn":
        # multi-scale conv front-end (causal)
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
        x = layers.Conv1D(64, 5, padding="causal", activation="gelu", dilation_rate=2)(x)
        x = layers.Dropout(0.25)(x)
        # BiLSTM backbone
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Bidirectional(layers.LSTM(64,  return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        # self-attention residual
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="gelu")(x)
        x = layers.Dropout(0.25)(x)

    elif variant == "tcn_bilstm_attn":
        # Temporal ConvNet-ish stack
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=1)(inp)
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=2)(x)
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=4)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.3, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = layers.Add()([x, attn]); x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128, activation="gelu")(x)
        x = layers.Dropout(0.3)(x)

    else:
        raise ValueError(f"Unknown MODEL_VARIANT: {variant}")

    out = layers.Dense(PRED_STEPS)(x)
    model = models.Model(inp, out)

    # cosine decay restarts + AdamW
    steps_in_cycle = STEPS_PER_EPOCH * 8  # ~8 epochs/cycle
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=steps_in_cycle,
        t_mul=2.0, m_mul=0.8, alpha=1e-5
    )
    opt = optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=losses.Huber(delta=1.0), metrics=["mae"])
    return model

model = build_model(MODEL_VARIANT)
print(model.summary())

ckpt_path = os.path.join(OUT_DIR, "best_lstm.keras")
cbs = [
    callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
    callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VAL_STEPS,
    verbose=1,
    callbacks=cbs
)

# training plots
plt.figure(); plt.plot(history.history["loss"], label="train")
plt.plot(history.history.get("val_loss", []), label="val"); plt.legend(); plt.grid(True)
plt.title("Training – Huber loss"); savefig(figpath("train_loss.png"))
if "mae" in history.history:
    plt.figure(); plt.plot(history.history["mae"], label="train")
    if "val_mae" in history.history: plt.plot(history.history["val_mae"], label="val")
    plt.legend(); plt.grid(True); plt.title("Training – MAE"); savefig(figpath("train_mae.png"))

# ====================== EVALUATION ======================
y_pred = model.predict(X_test, verbose=0)
means_s = means[pidx_test].reshape(-1,1); stds_s = stds[pidx_test].reshape(-1,1)
if TARGET_MODE=="delta":
    x_last_norm = ts_norm[pidx_test, tidx_test-1].reshape(-1,1)
    last_den    = x_last_norm * stds_s + means_s
    pred_den    = last_den + (y_pred * stds_s)
    true_den    = last_den + (y_test * stds_s)
else:
    pred_den    = y_pred * stds_s + means_s
    true_den    = y_test * stds_s + means_s

mae = mean_absolute_error(true_den.ravel(), pred_den.ravel())
rmse = math.sqrt(mean_squared_error(true_den.ravel(), pred_den.ravel()))
r2   = r2_score(true_den.ravel(), pred_den.ravel())
print(f"[RESULT] Test (sampled {len(true_den):,})  MAE: {mae:.3f}  RMSE: {rmse:.3f}  R2: {r2:.3f}")

# per-point forecast (next step) using the same transforms
X_fore = np.stack([build_window_channels(ts_norm[i, -LOOKBACK:]) for i in range(n_points)], axis=0)
y_next_norm = model.predict(X_fore, verbose=0)
if TARGET_MODE=="delta":
    last_all = ts_norm[:, -1].reshape(-1,1) * stds + means
    y_next_den = last_all + (y_next_norm * stds)
else:
    y_next_den = y_next_norm * stds + means
df["pred_next_step"] = y_next_den[:,0]
pred_csv = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(CSV_PATH))[0] + "_with_prediction.csv")
df.to_csv(pred_csv, index=False)
print("[SAVED] Predictions CSV:", pred_csv)

# ====================== VISUALIZATION SUITE (ALL) ======================
# 1) VEL histogram
if "VEL" in df.columns:
    plt.figure(); plt.hist(df["VEL"].dropna().values, bins=60)
    plt.title("Distribution of VEL"); plt.xlabel("VEL"); plt.ylabel("Count"); plt.grid(True)
    savefig(figpath("vel_hist.png"))

# 2) Missingness heatmap (sampled points)
mask = ts_orig.isna().astype(int)
n = min(MISSING_SAMPLE_N, mask.shape[0]); sample_idx = np.random.choice(mask.shape[0], n, replace=False)
plt.figure(figsize=(10,4)); plt.imshow(mask.iloc[sample_idx].to_numpy(), aspect="auto", interpolation="nearest")
plt.title("Missingness Heatmap (1=NaN) – sampled points"); plt.xlabel("Time"); plt.ylabel("Points")
plt.colorbar(); savefig(figpath("missingness_heatmap.png"))

# 3) Global mean + IQR
mean_t = np.nanmean(arr_vals, axis=0)
q25_t  = np.nanpercentile(arr_vals, 25, axis=0)
q75_t  = np.nanpercentile(arr_vals, 75, axis=0)
plt.figure(); plt.plot(date_index, mean_t, label="Mean")
plt.fill_between(date_index, q25_t, q75_t, alpha=0.3, label="IQR (25–75%)")
plt.title("Global Cumulative Deformation (Mean + IQR)"); plt.legend(); plt.grid(True)
plt.gcf().autofmt_xdate(); savefig(figpath("global_mean_iqr.png"))

# 4) Temporal variance
var_t = np.var(arr_vals, axis=0)
plt.figure(); plt.plot(date_index, var_t); plt.title("Temporal Variance across points"); plt.grid(True)
plt.gcf().autofmt_xdate(); savefig(figpath("temporal_variance.png"))

# 5) Time-step correlation (subset)
step = max(1, len(date_cols)//MAX_CORR_STEPS)
subset_cols = date_cols[::step][:MAX_CORR_STEPS]
subset_vals = df[subset_cols].ffill(axis=1).bfill(axis=1).to_numpy(np.float32)
corr_t = np.corrcoef(subset_vals.T)
plt.figure(figsize=(6,5)); plt.imshow(corr_t, aspect="auto", interpolation="nearest")
plt.title("Correlation Between Time Steps (subset)"); plt.colorbar()
savefig(figpath("time_correlation.png"))

# 6) Spatial maps
def scatter_map(z, title, fname, cb="Value"):
    plt.figure(figsize=(6,5))
    sc = plt.scatter(lons, lats, c=z, s=4)
    plt.colorbar(sc, label=cb); plt.title(title); plt.xlabel("Lon"); plt.ylabel("Lat"); plt.grid(True)
    savefig(figpath(fname))

last_def = df[date_cols[-1]].values
scatter_map(last_def, f"Last Observed Cumulative Deformation ({date_cols[-1]})", "spatial_last.png", "Deformation")
if "VEL" in df.columns:
    scatter_map(df["VEL"].values, "Velocity (VEL)", "spatial_vel.png", "VEL")
scatter_map(df["pred_next_step"].values, "Forecasted Next-Step Deformation", "spatial_pred_next.png", "Predicted")

# 7) Top-|VEL| time series
if "VEL" in df.columns:
    k = min(TOP_VEL_K, len(df))
    top_idx = np.argsort(np.abs(df["VEL"].values))[-k:]
    for idx in top_idx:
        y = ts_clean.iloc[idx].values
        plt.figure(); plt.plot(date_index, y); plt.title(f"Point #{idx} – Time Series (Top |VEL|)")
        plt.grid(True); plt.gcf().autofmt_xdate(); savefig(figpath(f"ts_point_{idx}.png"))

# 8) PCA across points + cluster trends
arr_std = (arr_vals - arr_vals.mean(axis=1, keepdims=True)) / (arr_vals.std(axis=1, keepdims=True)+1e-8)
pca = PCA(n_components=2, random_state=0)
pca_sample = min(150_000, n_points)
sel = np.random.choice(n_points, pca_sample, replace=False)
pc = pca.fit_transform(arr_std[sel])

plt.figure(figsize=(6,5))
c = df["VEL"].values[sel] if "VEL" in df.columns else last_def[sel]
sc = plt.scatter(pc[:,0], pc[:,1], c=c, s=4); plt.colorbar(sc)
plt.title("PCA of Point Time Series (colored by VEL or last deformation)")
savefig(figpath("pca_points.png"))

if RUN_KMEANS:
    # NOTE: this KMeans is for visualization only (cluster trend plots).
    # For the paper's Table 10 and Figure 14, use cluster_labels_all.npy
    # produced by fix_cluster_assignment.py (unified assignment for all points).
    km = KMeans(n_clusters=KMEANS_CLUSTERS, n_init=10, random_state=0)
    labels = km.fit_predict(pc)
    plt.figure(figsize=(6,5)); plt.scatter(pc[:,0], pc[:,1], c=labels, s=4, cmap="tab10")
    plt.title(f"PCA colored by KMeans clusters (k={KMEANS_CLUSTERS})")
    savefig(figpath("pca_kmeans.png"))
    for c_id in range(KMEANS_CLUSTERS):
        idx = sel[labels==c_id]
        if len(idx)==0: continue
        Y = arr_vals[idx]
        m = np.nanmean(Y, axis=0); q25 = np.nanpercentile(Y,25,axis=0); q75=np.nanpercentile(Y,75,axis=0)
        plt.figure(); plt.plot(date_index, m, label="Mean")
        plt.fill_between(date_index, q25, q75, alpha=0.3, label="IQR"); plt.legend(); plt.grid(True)
        plt.title(f"Cluster {c_id}: Deformation Trend (n={len(idx)})")
        plt.gcf().autofmt_xdate(); savefig(figpath(f"cluster_{c_id}_trend.png"))

# 9) Test diagnostics
plt.figure(); 
plt.scatter(true_den.ravel()[::50], pred_den.ravel()[::50], s=4, alpha=0.5)
mn = min(true_den.min(), pred_den.min()); mx = max(true_den.max(), pred_den.max())
plt.plot([mn,mx],[mn,mx],'k--',linewidth=1); plt.grid(True)
plt.title("True vs Predicted (Test, denormalized)"); plt.xlabel("True"); plt.ylabel("Predicted")
savefig(figpath("test_true_vs_pred.png"))

err = (pred_den - true_den).ravel()
plt.figure(); plt.hist(err, bins=80); plt.grid(True)
plt.title("Prediction Error Distribution (Test)"); plt.xlabel("Pred - True")
savefig(figpath("test_error_hist.png"))

# 10) Per-point MAE + spatial map + worst-point timeline
point_mae = np.zeros(n_points, np.float64); counts = np.zeros(n_points, np.int64)
ae = np.abs((pred_den - true_den).ravel())
for i,p in enumerate(pidx_test):
    point_mae[p] += ae[i]; counts[p]+=1
counts[counts==0]=1; point_mae /= counts
scatter_map(point_mae, "Per-Point MAE on Test Horizon", "spatial_mae.png", "MAE")

# worst point: plot true vs pred values at test times on top of context series
worst_point = int(np.argmax(point_mae))
t_low, t_high = TE_RANGE
times = np.arange(t_low, t_high)
Xp = np.stack([build_window_channels(ts_norm[worst_point, t-LOOKBACK:t]) for t in times], axis=0)
yp = model.predict(Xp, verbose=0)  # normalized delta/level
last_norm = ts_norm[worst_point, times-1].reshape(-1,1)
last_den  = last_norm * stds[worst_point] + means[worst_point]
if TARGET_MODE=="delta":
    pred_series = (last_den + yp * stds[worst_point])[:,0]
    true_series = ts_clean.iloc[worst_point, times].astype(float).values  # observed level
else:
    pred_series = (yp * stds[worst_point] + means[worst_point])[:,0]
    true_series = ts_clean.iloc[worst_point, times].astype(float).values
raw = ts_clean.iloc[worst_point].values
plt.figure()
ctx_len = min(200, len(raw))
plt.plot(date_index[-ctx_len:], raw[-ctx_len:], label="Observed (context)")
plt.scatter(pd.to_datetime([date_cols[t] for t in times]), true_series, s=10, label="True (test)")
plt.scatter(pd.to_datetime([date_cols[t] for t in times]), pred_series, s=10, label="Pred (test)")
plt.title(f"Point #{worst_point}: Context + Test Targets vs Predictions"); plt.legend(); plt.grid(True)
plt.gcf().autofmt_xdate(); savefig(figpath("worst_point_timeline.png"))

# 11) Last vs forecasted next step
plt.figure(); plt.scatter(df[date_cols[-1]].values, df["pred_next_step"].values, s=6, alpha=0.4)
lo = min(df[date_cols[-1]].min(), df["pred_next_step"].min())
hi = max(df[date_cols[-1]].max(), df["pred_next_step"].max())
plt.plot([lo,hi],[lo,hi],'k--'); plt.grid(True)
plt.title("Last Observed vs Forecasted Next Step")
plt.xlabel(f"Observed @ {date_cols[-1]}"); plt.ylabel("Forecasted next step")
savefig(figpath("last_vs_forecast.png"))

# 12) Animation (GIF; MP4 optional)
def animate_temporal_map(lons, lats, data, dates, every_nth=3, gif_name="deformation_map.gif", mp4_name="deformation_map.mp4"):
    from matplotlib import animation
    idx = list(range(0, data.shape[1], every_nth))
    if idx[-1] != data.shape[1]-1: idx.append(data.shape[1]-1)
    vmin = float(np.nanpercentile(data[:, idx], 5)); vmax = float(np.nanpercentile(data[:, idx], 95))
    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(lons, lats, c=data[:, idx[0]], s=2, vmin=vmin, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Cumulative deformation")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True)
    def update(i):
        t = idx[i]; sc.set_array(data[:, t]); ax.set_title(f"Deformation @ {dates[t]}"); return sc,
    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=200, blit=False)
    try:
        from matplotlib.animation import PillowWriter
        anim.save(os.path.join(OUT_DIR, gif_name), writer=PillowWriter(fps=6)); print("[SAVED] Animation GIF:", os.path.join(OUT_DIR, gif_name))
    except Exception as e: print("[WARN] GIF save failed:", e)
    try:
        from matplotlib.animation import FFMpegWriter
        anim.save(os.path.join(OUT_DIR, mp4_name), writer=FFMpegWriter(fps=6, bitrate=1800)); print("[SAVED] MP4:", os.path.join(OUT_DIR, mp4_name))
    except Exception as e: print("[WARN] MP4 save skipped:", e)
    plt.close(fig)

animate_temporal_map(lons, lats, arr_vals, date_cols, every_nth=3)

# 13) Folium (HeatMap + sampled markers) — HTML that opens fast
def folium_maps():
    center = [float(np.nanmean(lats)), float(np.nanmean(lons))]
    # HeatMap (aggregated)
    hm_map = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")
    v = df[date_cols[-1]].values.astype(float)
    sel = np.random.choice(n_points, min(200_000, n_points), replace=False)
    heat_data = np.column_stack([lats[sel], lons[sel], v[sel]]).tolist()
    plugins.HeatMap(heat_data, radius=8, blur=12, max_zoom=12).add_to(hm_map)
    hm_path = os.path.join(OUT_DIR, "deformation_heatmap.html"); hm_map.save(hm_path)
    # FastMarkerCluster sampled
    mk_map = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")
    sel2 = np.random.choice(n_points, min(FOLIUM_SAMPLE, n_points), replace=False)
    points = np.column_stack([lats[sel2], lons[sel2]]).tolist()
    plugins.FastMarkerCluster(points).add_to(mk_map)
    mk_path = os.path.join(OUT_DIR, "deformation_markers_sampled.html"); mk_map.save(mk_path)
    return hm_path, mk_path

hm_path, mk_path = folium_maps()
print("[SAVED] Interactive heatmap:", hm_path)
print("[SAVED] Interactive sampled markers:", mk_path)

# ====================== XAI ======================
@tf.function
def grad_times_input(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x, training=False)[:,0]
    return tape.gradient(y, x) * x   # (B,L,C)

def integrated_gradients(x, steps=64, baseline=None):
    x = tf.convert_to_tensor(x)
    if baseline is None: baseline = tf.zeros_like(x)
    alphas = tf.linspace(0., 1., steps+1)[1:]
    total = tf.zeros_like(x)
    for a in alphas:
        xi = baseline + a*(x-baseline)
        with tf.GradientTape() as tape:
            tape.watch(xi); y = model(xi, training=False)[:,0]
        total += tape.gradient(y, xi)
    return (x-baseline) * (total/steps)  # (B,L,C)

def smooth_ig(x, n=6, noise_std=0.05, steps=32):
    acc = np.zeros_like(x.numpy())
    for _ in range(n):
        noise = np.random.normal(0, noise_std, size=x.shape).astype(x.dtype.as_numpy_dtype)
        acc += integrated_gradients(x + noise, steps=steps).numpy()
    return acc / n

def collapse_channels(attr):  # (B,L,C) -> (B,L)
    return np.sum(np.abs(attr), axis=2)

B = min(XAI_SAMPLES, X_test.shape[0])
X_sub = tf.convert_to_tensor(X_test[:B])
y_sub = y_test[:B]
gxi_full = grad_times_input(X_sub).numpy()
ig_full  = integrated_gradients(X_sub, steps=64).numpy()
sig_full = smooth_ig(X_sub, n=6, noise_std=0.05, steps=32)

gxi = collapse_channels(gxi_full)
ig  = collapse_channels(ig_full)
sig = collapse_channels(sig_full)

def bar_global(arr, title, fname):
    vals = np.mean(np.abs(arr), axis=0)
    plt.figure(figsize=(8,3)); plt.bar(range(LOOKBACK), vals)
    plt.xticks(range(LOOKBACK), [f"t-{LOOKBACK - j}" for j in range(LOOKBACK)])
    plt.title(title); plt.grid(axis="y", alpha=.3); savefig(figpath(fname))

bar_global(gxi, "Global lag importance – Grad×Input", "xai_global_gradxin.png")
bar_global(ig,  "Global lag importance – Integrated Gradients", "xai_global_ig.png")
bar_global(sig, "Global lag importance – SmoothGrad-IG", "xai_global_smoothig.png")

# Local IG heatmap
plt.figure(figsize=(10,1.6)); plt.imshow(ig[0:1,:], aspect="auto", cmap="viridis")
plt.yticks([]); plt.xticks(range(LOOKBACK), [f"t-{LOOKBACK - j}" for j in range(LOOKBACK)])
plt.title("Local IG over lags"); plt.colorbar(fraction=0.046, pad=0.04)
savefig(figpath("xai_local_ig.png"))

# Deletion/Insertion (multichannel-aware)
def deletion_insertion_curves(X, y, attr_lag, prefix):
    order = np.argsort(-np.abs(attr_lag), axis=1)  # per-sample lag order
    Xd = X.numpy().copy()
    Xi = np.zeros_like(Xd)
    perc=[0.0]
    def mse(yhat): return mean_squared_error(y[:,0], yhat[:,0])
    errD=[mse(model.predict(Xd, verbose=0))]
    errI=[mse(model.predict(Xi, verbose=0))]
    for m in range(1, LOOKBACK+1):
        for i in range(Xd.shape[0]):
            l = order[i, m-1]
            Xd[i, l, :] = 0.0
            Xi[i, l, :] = X.numpy()[i, l, :]
        ed = mse(model.predict(Xd, verbose=0))
        ei = mse(model.predict(Xi, verbose=0))
        perc.append(m/LOOKBACK); errD.append(ed); errI.append(ei)
    perc = np.array(perc)
    aucD = trapezoid(np.array(errD), perc) / (perc[-1]-perc[0])
    aucI = trapezoid(np.array(errI), perc) / (perc[-1]-perc[0])
    plt.figure(); plt.plot(perc*100, errD); plt.grid(True)
    plt.xlabel("% lags deleted"); plt.ylabel("MSE"); plt.title(f"Deletion – {prefix} (AUC={aucD:.3f})")
    savefig(figpath(f"xai_deletion_{prefix}.png"))
    plt.figure(); plt.plot(perc*100, errI); plt.grid(True)
    plt.xlabel("% lags inserted"); plt.ylabel("MSE"); plt.title(f"Insertion – {prefix} (AUC={aucI:.3f})")
    savefig(figpath(f"xai_insertion_{prefix}.png"))
    return aucD, aucI

auc_gx_d, auc_gx_i = deletion_insertion_curves(X_sub, y_sub, gxi, "GradXInput")
auc_ig_d,  auc_ig_i  = deletion_insertion_curves(X_sub, y_sub, ig,  "IG")
auc_si_d,  auc_si_i  = deletion_insertion_curves(X_sub, y_sub, sig, "SmoothIG")

# Permutation importance per lag (permute all channels at that lag)
def permutation_importance(model, X, y, trials=3):
    base = mean_squared_error(y[:,0], model.predict(X, verbose=0)[:,0])
    L = X.shape[1]; scores = np.zeros(L, float)
    for l in range(L):
        inc=0.0
        for _ in range(trials):
            Xp = X.numpy().copy()
            perm = np.random.permutation(Xp.shape[0])
            Xp[:, l, :] = Xp[perm, l, :]
            inc += mean_squared_error(y[:,0], model.predict(Xp, verbose=0)[:,0]) - base
        scores[l] = inc/trials
    return scores
perm_scores = permutation_importance(model, X_sub, y_sub, trials=PERM_TRIALS)
bar_global(perm_scores[None,:], "Permutation Importance (ΔMSE per lag)", "xai_global_permutation.png")

# Spatial: reliance on recent lags (batched IG over all points; multichannel)
def spatial_recent_reliance(k=RECENT_K, batch=8192, steps=32):
    X_all = np.stack([build_window_channels(ts_norm[i, -LOOKBACK:]) for i in range(n_points)], axis=0)
    n = X_all.shape[0]; out = np.zeros(n, np.float32)
    for s in range(0, n, batch):
        xb = tf.convert_to_tensor(X_all[s:s+batch])
        igb = integrated_gradients(xb, steps=steps).numpy()  # (B,L,C)
        out[s:s+batch] = np.mean(np.sum(np.abs(igb[:, -k:, :]), axis=2), axis=1)
    return out

recent_attr = spatial_recent_reliance()
plt.figure(figsize=(6,5))
sc = plt.scatter(lons, lats, c=recent_attr, s=3); plt.colorbar(sc, label=f"mean |IG| (last {RECENT_K} lags, all channels)")
plt.title(f"Spatial reliance on recent lags (k={RECENT_K})")
savefig(figpath("xai_spatial_recent.png"))

# Folium HeatMap for XAI
xai_map = folium.Map(location=[float(np.nanmean(lats)), float(np.nanmean(lons))], zoom_start=9, tiles="OpenStreetMap")
selx = np.random.choice(n_points, min(200_000, n_points), replace=False)
heat_xai = np.column_stack([lats[selx], lons[selx], recent_attr[selx]]).tolist()
plugins.HeatMap(heat_xai, radius=8, blur=12, max_zoom=12).add_to(xai_map)
xai_html = os.path.join(OUT_DIR, "xai_spatial_recent.html"); xai_map.save(xai_html)
print("[SAVED] XAI spatial HeatMap:", xai_html)

# XAI tables
pd.DataFrame({
    "method":["Grad×Input","IG","SmoothIG"],
    "deletion_auc":[auc_gx_d, auc_ig_d, auc_si_d],
    "insertion_auc":[auc_gx_i, auc_ig_i, auc_si_i],
}).to_csv(os.path.join(OUT_DIR,"xai_faithfulness_auc.csv"), index=False)
np.savetxt(os.path.join(OUT_DIR,"xai_perm_scores.csv"), np.array([perm_scores]), delimiter=",", fmt="%.6f")

def topk_names(arr, k=5):
    order = np.argsort(-arr)[:k]
    return [f"t-{LOOKBACK - j}" for j in order]
pd.DataFrame({
    "Grad×Input": [", ".join(topk_names(np.mean(np.abs(gxi),0)))],
    "IG":         [", ".join(topk_names(np.mean(np.abs(ig),0)))],
    "SmoothIG":   [", ".join(topk_names(np.mean(np.abs(sig),0)))],
    "Permutation":[", ".join(topk_names(perm_scores))]
}).to_csv(os.path.join(OUT_DIR, "xai_topk_lags.csv"), index=False)

# optional SHAP (flatten L*C features)
if ENABLE_SHAP:
    try:
        _pip_install(["shap"])
        import shap
        def flattener(X): return X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
        def predict_from_flat(Xflat):
            Xr = Xflat.reshape((-1, LOOKBACK, CHANNELS)); return model.predict(Xr, verbose=0)[:,0]
        BKG = min(200, X_test.shape[0]); EXS = min(150, X_test.shape[0])
        bg_idx = np.random.choice(X_test.shape[0], BKG, replace=False)
        ex_idx = np.random.choice(X_test.shape[0], EXS, replace=False)
        X_bg, X_ex = flattener(X_test[bg_idx]), flattener(X_test[ex_idx])
        expl = shap.KernelExplainer(predict_from_flat, X_bg, link="identity")
        shap_vals = expl.shap_values(X_ex, nsamples="auto")
        shap.summary_plot(shap_vals, X_ex, feature_names=[f"{c}/t-{LOOKBACK - j}" for j in range(LOOKBACK) for c in (["x","dx"] if USE_SLOPE_CHANNEL else ["x"])], show=False)
        savefig(figpath("xai_shap_beeswarm.png"))
    except Exception as e:
        print("[WARN] SHAP skipped:", e)

# ====================== REPORT ======================
report = os.path.join(OUT_DIR, "report.html")
with open(report, "w", encoding="utf-8") as f:
    f.write(f"""
<html><head><meta charset="utf-8"><title>Temporal Deformation – LSTM Results</title></head>
<body style="font-family:Arial,sans-serif;margin:20px">
<h1>Temporal Deformation – LSTM/Hybrid Results</h1>
<p><b>Test MAE:</b> {mae:.3f} &nbsp; <b>RMSE:</b> {rmse:.3f} &nbsp; <b>R2:</b> {r2:.3f}
&nbsp; <b>Target:</b> {TARGET_MODE} &nbsp; <b>Model:</b> {MODEL_VARIANT} &nbsp; <b>LOOKBACK:</b> {LOOKBACK}
&nbsp; <b>Channels:</b> {CHANNELS} (detrend={DETREND_WINDOWS}, slope={USE_SLOPE_CHANNEL})</p>

<h2>Training</h2>
<img src="train_loss.png" style="max-width:900px"><br>
<img src="train_mae.png" style="max-width:900px"><br>

<h2>Interactive</h2>
<p><a href="deformation_heatmap.html">Interactive HeatMap (last date)</a> &nbsp; | &nbsp; <a href="deformation_markers_sampled.html">Interactive markers (sampled)</a></p>
<p><a href="xai_spatial_recent.html">XAI: reliance on recent lags (HeatMap)</a></p>
<p>Animation GIF:</p><img src="deformation_map.gif" style="max-width:900px"><br>

<h2>Spatial (PNG)</h2>
<img src="spatial_last.png" style="max-width:900px"><br>
{"<img src='spatial_vel.png' style='max-width:900px'><br>" if "VEL" in df.columns else ""}
<img src="spatial_pred_next.png" style="max-width:900px"><br>
<img src="spatial_mae.png" style="max-width:900px"><br>
<img src="xai_spatial_recent.png" style="max-width:900px"><br>

<h2>Temporal + Stats</h2>
<img src="global_mean_iqr.png" style="max-width:900px"><br>
<img src="temporal_variance.png" style="max-width:900px"><br>
<img src="time_correlation.png" style="max-width:900px"><br>
<img src="missingness_heatmap.png" style="max-width:900px"><br>
<img src="test_true_vs_pred.png" style="max-width:900px"><br>
<img src="test_error_hist.png" style="max-width:900px"><br>
<img src="last_vs_forecast.png" style="max-width:900px"><br>
{"".join([f"<img src='ts_point_{idx}.png' style='max-width:900px'><br>" for idx in (np.argsort(np.abs(df['VEL'].values))[-min(TOP_VEL_K,len(df)):] if 'VEL' in df.columns else [])])}
<img src="pca_points.png" style="max-width:900px"><br>
{"<img src='pca_kmeans.png' style='max-width:900px'><br>" if RUN_KMEANS else ""}
{"".join([f"<img src='cluster_{c}_trend.png' style='max-width:900px'><br>" for c in range(KMEANS_CLUSTERS)])}
<img src="worst_point_timeline.png" style="max-width:900px"><br>

<h2>XAI – Global</h2>
<img src="xai_global_gradxin.png" style="max-width:900px"><br>
<img src="xai_global_ig.png" style="max-width:900px"><br>
<img src="xai_global_smoothig.png" style="max-width:900px"><br>
<img src="xai_global_permutation.png" style="max-width:900px"><br>
{"<img src='xai_shap_beeswarm.png' style='max-width:900px'><br>" if os.path.exists(figpath("xai_shap_beeswarm.png")) else ""}
<p><a href="xai_faithfulness_auc.csv">faithfulness AUC (CSV)</a> &nbsp; | &nbsp; <a href="xai_topk_lags.csv">top-k lags (CSV)</a> &nbsp; | &nbsp; <a href="xai_perm_scores.csv">perm scores (CSV)</a></p>

<h2>XAI – Local</h2>
<img src="xai_local_ig.png" style="max-width:900px"><br>

</body></html>
""")
print("[SAVED] Report:", report)
print("\n[DONE] Pipeline completed.")
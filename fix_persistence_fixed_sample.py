# -*- coding: utf-8 -*-
"""
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Landslide Deformation using EGMS Time Series"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing 

fix_persistence_fixed_sample.py
================================
Recomputes the Persistence baseline on the EXACT same fixed 200,000-window
test sample used by the proposed model (seed=42, p=2.0, same TE_RANGE).

This resolves the discrepancy between Table 5 (5.234 mm) and Table 8 (3.224 mm):
both must use the identical sample so the numbers are directly comparable.

After running this script:
  - Use the NEW Persistence MAE/RMSE/R² in BOTH Table 5 AND Table 8.
  - Remove the caption note about "sampling variance across draws."

Output: fix_persistence_results.csv
"""

import os, re, math
import numpy as np
import pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
tf.random.set_seed(SEED)   # set after TF import for full reproducibility
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ====================== EDIT THESE PATHS IF NEEDED ======================
CSV_PATH   = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
MODEL_PATH = r"E:\خروجی های پروژه\outputs_LSTM_BiLSTM\outputs\best_lstm.keras"
OUT_DIR    = os.path.dirname(MODEL_PATH)
# ========================================================================

# Must match the original config EXACTLY
LOOKBACK    = 24
CHANNELS    = 2
DETREND_WINDOWS   = True
USE_SLOPE_CHANNEL = True
WINSOR_X    = 6.0
BIAS_POWER  = 2.0
SEED        = 42        # same seed used for Table 5 / main evaluation
N_TEST      = 200_000   # same sample size

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
print(f"[INFO] n_points={n_points:,}, T={T}, val_end={val_end}")

# Per-point train-only normalization
train_obs = ts_ff.iloc[:, :train_end].to_numpy(np.float32)
means = np.nanmean(train_obs, axis=1, keepdims=True)
stds  = np.nanstd(train_obs,  axis=1, keepdims=True)
means = np.nan_to_num(means, nan=0.0)
stds  = np.nan_to_num(stds,  nan=1.0)
stds[stds == 0] = 1.0
ts_norm = ((arr_vals - means) / stds).astype(np.float32)
TE_RANGE = (val_end, T)

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

def build_fixed_sample(n, bias_power, seed):
    rng = np.random.default_rng(seed)
    t_low, t_high = TE_RANGE
    if bias_power == 1.0:
        t = rng.integers(t_low, t_high, size=n)
    else:
        u = rng.random(n) ** bias_power
        t = (t_low + (t_high - t_low - 1) * u).astype(int)
    p = rng.integers(0, n_points, size=n)
    X = np.zeros((n, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros(n, np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        w = ts_norm[pi, ti - LOOKBACK:ti]
        X[i] = build_window_channels(w)
        y[i] = ts_norm[pi, ti] - ts_norm[pi, ti - 1]
    return X, y, p, t

print(f"[INFO] Building fixed test sample (n={N_TEST:,}, seed={SEED}, p={BIAS_POWER}) ...")
X_test, y_norm, pidx, tidx = build_fixed_sample(N_TEST, BIAS_POWER, SEED)

stds_s  = stds[pidx].reshape(-1)
means_s = means[pidx].reshape(-1)
x_last_norm = ts_norm[pidx, tidx - 1]
last_den    = x_last_norm * stds_s + means_s
true_delta_den = y_norm * stds_s
true_level_den = last_den + true_delta_den

# ── Proposed model ──────────────────────────────────────────────────────
print(f"[INFO] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
pred_norm = model.predict(X_test, verbose=0, batch_size=4096).ravel()
pred_delta_den_model = pred_norm * stds_s
pred_level_den_model = last_den + pred_delta_den_model

mae_model  = mean_absolute_error(true_level_den, pred_level_den_model)
rmse_model = math.sqrt(mean_squared_error(true_level_den, pred_level_den_model))
r2_model   = r2_score(true_level_den, pred_level_den_model)
r2d_model  = r2_score(true_delta_den, pred_delta_den_model)

# ── Persistence / zero-increment ────────────────────────────────────────
# predicted delta = 0  →  predicted level = last observed
pred_level_den_pers = last_den.copy()
pred_delta_den_pers = np.zeros_like(true_delta_den)

mae_pers  = mean_absolute_error(true_level_den, pred_level_den_pers)
rmse_pers = math.sqrt(mean_squared_error(true_level_den, pred_level_den_pers))
r2_pers   = r2_score(true_level_den, pred_level_den_pers)
r2d_pers  = r2_score(true_delta_den, pred_delta_den_pers)
delta_mean = float(np.mean(true_delta_den))

print("\n" + "=" * 65)
print("RESULTS ON THE EXACT FIXED 200k SAMPLE (seed=42, p=2.0)")
print("=" * 65)
print(f"\nProposed CNN-BiLSTM-Attention:")
print(f"  MAE={mae_model:.3f}  RMSE={rmse_model:.3f}  R²={r2_model:.3f}  R²_delta={r2d_model:.4f}")
print(f"\nPersistence / zero-increment:")
print(f"  MAE={mae_pers:.3f}  RMSE={rmse_pers:.3f}  R²={r2_pers:.3f}  R²_delta={r2d_pers:.4f}")
print(f"  delta_mean = {delta_mean:.4f} mm")
print(f"\n{'='*65}")
print(f"USE THESE NUMBERS IN BOTH TABLE 5 AND TABLE 8:")
print(f"  Persistence: MAE={mae_pers:.3f}, RMSE={rmse_pers:.3f}, R²={r2_pers:.3f}")
print(f"{'='*65}")

rows = [
    {"Model": "CNN-BiLSTM-Attention (proposed)",
     "MAE": round(mae_model,3), "RMSE": round(rmse_model,3),
     "R2_level": round(r2_model,3), "R2_delta": round(r2d_model,4)},
    {"Model": "Persistence / zero-increment",
     "MAE": round(mae_pers,3), "RMSE": round(rmse_pers,3),
     "R2_level": round(r2_pers,3), "R2_delta": round(r2d_pers,4)},
]
df_out = pd.DataFrame(rows)
out_path = os.path.join(OUT_DIR, "fix_persistence_results.csv")
df_out.to_csv(out_path, index=False)
print(f"\n[SAVED] {out_path}")
print("\nDONE.")

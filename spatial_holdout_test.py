# -*- coding: utf-8 -*-
"""
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Landslide Deformation using EGMS Time Series"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing 

Standalone SPATIAL HOLD-OUT generalization test (Item 7).

Concern (R1, R4, advisor): the model has only been evaluated on a temporal
split (train/val/test all drawn from the SAME spatial corridor). This does
not demonstrate that the model generalises to UNSEEN locations -- it could,
in principle, be partially memorising point-specific behaviour.

What this script does (NO retraining -- uses the already-trained model):
  1) Splits the 594,978 points into a SPATIAL hold-out set (a geographic
     block, e.g. the eastern ~20% of the corridor by longitude) and the
     remaining "interior" points.
  2) Builds the SAME fixed-size test sample (same temporal test interval,
     same delta-mode target, same preprocessing) but restricted to:
       (a) the spatial hold-out block only
       (b) the interior (non-held-out) points only, for a fair comparison
  3) Runs inference with the trained model (best_lstm.keras) on both,
     and reports MAE / RMSE / R2 (level-space, matching the paper's main
     metric) for each, plus a simple spatial scatter plot showing which
     points were held out.

If performance on the spatial hold-out block is close to the interior
performance, this is strong evidence the model is NOT overfitting to
specific point identities and generalises spatially within the corridor.

Outputs (in OUT_DIR):
  spatial_holdout_results.csv
  spatial_holdout_map.png

Run time: a few minutes (inference only).
"""

import os, re, math, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
tf.random.set_seed(SEED)   # set after TF import for full reproducibility
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ====================== EDIT THESE PATHS IF NEEDED ======================
CSV_PATH   = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
MODEL_PATH = r"E:\خروجی های پروژه\outputs_LSTM_BiLSTM\outputs\best_lstm.keras"
OUT_DIR    = os.path.dirname(MODEL_PATH)
# ==========================================================================

# ---- must match the original config exactly ----
LOOKBACK    = 24
CHANNELS    = 2
DETREND_WINDOWS   = True
USE_SLOPE_CHANNEL = True
WINSOR_X    = 6.0
BIAS_POWER  = 2.0   # same recency-biased sampling used in the paper's headline metric
SEED        = 42

N_PER_GROUP = 100_000   # test windows drawn from EACH group (holdout / interior)

# Fraction of points (by longitude) assigned to the spatial hold-out block.
# 0.20 -> the easternmost 20% of points (by longitude) are held out entirely.
HOLDOUT_FRACTION = 0.20

np.random.seed(SEED)


def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))


print("[INFO] Loading CSV ...")
df = pd.read_csv(CSV_PATH)

for col in ("latitude", "longitude"):
    if col not in df.columns:
        raise ValueError(f"Expected a '{col}' column in the CSV -- check column names.")

lons = df["longitude"].to_numpy(np.float64)
lats = df["latitude"].to_numpy(np.float64)

date_cols = sorted([c for c in df.columns if _is_date_col(c)])
ts = df[date_cols].copy()
ts_ff = ts.ffill(axis=1)
ts_clean = ts_ff.fillna(0.0).astype(np.float32)
arr_vals = ts_clean.to_numpy(np.float32)

n_points, T = ts_clean.shape
train_end = int(T * 0.70)
val_end   = int(T * 0.85)
print(f"[INFO] n_points={n_points:,}, timesteps={T}")

train_vals_obs = ts_ff.iloc[:, :train_end].to_numpy(np.float32)
means = np.nanmean(train_vals_obs, axis=1, keepdims=True)
stds  = np.nanstd(train_vals_obs, axis=1, keepdims=True)
means = np.nan_to_num(means, nan=0.0)
stds  = np.nan_to_num(stds, nan=1.0)
stds[stds == 0] = 1.0

ts_norm = ((arr_vals - means) / stds).astype(np.float32)
TE_RANGE = (val_end, T)


def build_window_channels(w_norm):
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
    Xw = np.stack(outs, axis=-1)
    if WINSOR_X is not None:
        Xw = np.clip(Xw, -float(WINSOR_X), float(WINSOR_X))
    return Xw


# ====================== Define the spatial hold-out block ======================
lon_threshold = np.quantile(lons, 1.0 - HOLDOUT_FRACTION)
holdout_mask = lons >= lon_threshold          # easternmost HOLDOUT_FRACTION of points
interior_mask = ~holdout_mask

holdout_idx  = np.where(holdout_mask)[0]
interior_idx = np.where(interior_mask)[0]

print(f"\n[SPATIAL SPLIT] longitude threshold = {lon_threshold:.5f}")
print(f"  Hold-out block : {len(holdout_idx):,} points "
      f"({100*len(holdout_idx)/n_points:.1f}% of corridor)")
print(f"  Interior       : {len(interior_idx):,} points "
      f"({100*len(interior_idx)/n_points:.1f}% of corridor)")

# Quick map of which points are held out
plt.figure(figsize=(6, 5))
plt.scatter(lons[interior_idx], lats[interior_idx], s=2, c="#999999", label="Interior (seen region)")
plt.scatter(lons[holdout_idx], lats[holdout_idx], s=2, c="#E8821E", label="Spatial hold-out block")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.legend(markerscale=4)
plt.title("Spatial hold-out split")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "spatial_holdout_map.png"), dpi=180)
plt.close()
print(f"[SAVED] {os.path.join(OUT_DIR, 'spatial_holdout_map.png')}")


def sample_times_powerlaw(rng, t_low, t_high, size, bias_power):
    if bias_power == 1.0:
        return rng.integers(t_low, t_high, size=size)
    u = rng.random(size) ** bias_power
    return (t_low + (t_high - t_low - 1) * u).astype(int)


def build_eval_sample(arr_norm, time_range, point_pool, n, bias_power, seed):
    rng = np.random.default_rng(seed)
    t_low, t_high = time_range
    t = sample_times_powerlaw(rng, t_low, t_high, n, bias_power)
    p = point_pool[rng.integers(0, len(point_pool), size=n)]

    X = np.zeros((n, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((n, 1), np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        w = arr_norm[pi, ti - LOOKBACK:ti]
        X[i, :, :] = build_window_channels(w)
        y[i, 0] = arr_norm[pi, ti] - arr_norm[pi, ti - 1]
    return X, y, p, t


def evaluate_group(model, arr_norm, time_range, point_pool, n, seed, label):
    print(f"\n[BUILDING SAMPLE] {label}  (n={n:,}, pool_size={len(point_pool):,})")
    t0 = time.perf_counter()
    X, y, pidx, tidx = build_eval_sample(arr_norm, time_range, point_pool, n, BIAS_POWER, seed)
    print(f"  sample built in {time.perf_counter()-t0:.1f}s")

    y_pred = model.predict(X, verbose=0, batch_size=4096)

    means_s = means[pidx].reshape(-1, 1)
    stds_s  = stds[pidx].reshape(-1, 1)

    x_last_norm = arr_norm[pidx, tidx - 1].reshape(-1, 1)
    last_den    = x_last_norm * stds_s + means_s

    pred_den = last_den + (y_pred * stds_s)
    true_den = last_den + (y * stds_s)

    mae  = mean_absolute_error(true_den.ravel(), pred_den.ravel())
    rmse = math.sqrt(mean_squared_error(true_den.ravel(), pred_den.ravel()))
    r2   = r2_score(true_den.ravel(), pred_den.ravel())

    print(f"  [LEVEL] MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    return {"group": label, "n_test_windows": n, "n_points_pool": len(point_pool),
            "MAE": mae, "RMSE": rmse, "R2": r2}


print(f"\n[INFO] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

results = []
results.append(evaluate_group(
    model, ts_norm, TE_RANGE, interior_idx, N_PER_GROUP, seed=SEED,
    label=f"Interior (seen region, {100*(1-HOLDOUT_FRACTION):.0f}% of corridor)"
))
results.append(evaluate_group(
    model, ts_norm, TE_RANGE, holdout_idx, N_PER_GROUP, seed=SEED + 1,
    label=f"Spatial hold-out (unseen region, {100*HOLDOUT_FRACTION:.0f}% of corridor)"
))

df_out = pd.DataFrame(results)
out_path = os.path.join(OUT_DIR, "spatial_holdout_results.csv")
df_out.to_csv(out_path, index=False)

print("\n" + "=" * 70)
print("SPATIAL HOLD-OUT SUMMARY")
print("=" * 70)
print(df_out.to_string(index=False))

# Simple relative-difference summary
interior_row = df_out[df_out["group"].str.startswith("Interior")].iloc[0]
holdout_row  = df_out[df_out["group"].str.startswith("Spatial")].iloc[0]
mae_pct  = 100 * (holdout_row["MAE"] - interior_row["MAE"]) / interior_row["MAE"]
rmse_pct = 100 * (holdout_row["RMSE"] - interior_row["RMSE"]) / interior_row["RMSE"]
r2_diff  = holdout_row["R2"] - interior_row["R2"]

print(f"\nRelative change (hold-out vs interior):")
print(f"  MAE:  {mae_pct:+.1f}%")
print(f"  RMSE: {rmse_pct:+.1f}%")
print(f"  R2:   {r2_diff:+.4f}")

print(f"\n[SAVED] {out_path}")
print("\nDONE.")

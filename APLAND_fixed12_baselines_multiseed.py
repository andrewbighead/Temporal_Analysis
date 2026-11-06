# -*- coding: utf-8 -*-
"""
APLAND_fixed12_baselines_multiseed.py
========================================================
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

Reviewer-ready comparison on the corrected fixed 12-day EGMS protocol.

Scientific safeguards:
  - reindexes mixed 6/12/24-day raw acquisitions to a fixed 12-day grid;
  - forward-fill is used for inputs only;
  - excludes a target unless both t and t-1 are raw observations;
  - uses corrected recent-biased sampling, 1-(1-u)**p;
  - uses one shared 200,000-window test sample for every model and seed;
  - trains Linear Regression on training data (never on the test set);
  - trains the proposed model under the same protocol for all three seeds;
  - reports increment-space and reconstructed-level metrics separately;
  - applies the DM test to epoch-aggregated squared losses with the
    Harvey-Leybourne-Newbold finite-sample correction.

Example:
  python APLAND_fixed12_baselines_multiseed.py --csv "data/EGMS_L2b.csv" \
      --output-dir "outputs/fixed12_baselines" --multi-seed
"""

import os, re, random, warnings, argparse, math, json, gc, time
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, models
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG — keep identical to LSTM_BiLSTM-M.py
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH       = None
OUT_DIR        = None

LOOKBACK       = 24
PRED_STEPS     = 1
TARGET_MODE    = "delta"

DETREND_WINDOWS   = True
USE_SLOPE_CHANNEL = True
WINSOR_X          = 6.0
LABEL_CLIP_SIGMA  = 3.0

EPOCHS            = 60
BATCH_SIZE        = 1024
STEPS_PER_EPOCH   = 600
VAL_STEPS         = 120
PATIENCE          = 10

RECENCY_BIASED    = True
BIAS_POWER        = 2.0

SEEDS             = [42, 123, 2024]
CHANNELS          = 1 + int(USE_SLOPE_CHANNEL)   # = 2

# Fixed test size — must match LSTM_BiLSTM-M.py so errors align
N_TEST         = 200_000
DM_SEED        = 42       # seed used to build the fixed test set
HORIZON_DAYS   = 12
TRAIN_FRACTION = 0.70
VAL_FRACTION   = 0.15

def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data loading — IDENTICAL to LSTM_BiLSTM-M.py
# ─────────────────────────────────────────────────────────────────────────────
def load_data(anchor_text=None):
    assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH, low_memory=False)
    date_cols = sorted([c for c in df.columns if _is_date_col(c)])
    raw_dates = pd.DatetimeIndex(pd.to_datetime(date_cols, format="%Y%m%d"))
    order = np.argsort(raw_dates.values)
    raw_dates = raw_dates[order]
    date_cols = [date_cols[i] for i in order]
    deltas = np.diff(raw_dates.values).astype("timedelta64[D]").astype(int)
    anchor = pd.to_datetime(anchor_text, format="%Y%m%d") if anchor_text else raw_dates[0]
    if anchor not in raw_dates:
        raise ValueError(f"Anchor {anchor.date()} is not a raw acquisition date")
    count = int((raw_dates[-1] - anchor).days // HORIZON_DAYS) + 1
    grid = pd.date_range(anchor, periods=count, freq=f"{HORIZON_DAYS}D")
    raw_set = set(raw_dates)
    observed = np.asarray([d in raw_set for d in grid], dtype=bool)
    ts = df[date_cols].copy()
    ts.columns = raw_dates
    fixed = ts.reindex(columns=grid)
    missing_cells = int(fixed.isna().to_numpy().sum())
    arr_vals = fixed.ffill(axis=1).fillna(0.0).to_numpy(np.float32)
    n_points, T = arr_vals.shape
    train_end = int(T * 0.70); val_end = int(T * 0.85)
    train_obs = arr_vals[:, :train_end]
    means = np.mean(train_obs, axis=1, keepdims=True)
    stds  = np.std(train_obs, axis=1, keepdims=True)
    means = np.nan_to_num(means, nan=0.0)
    stds  = np.nan_to_num(stds,  nan=1.0)
    stds[stds == 0] = 1.0
    ts_norm = ((arr_vals - means) / stds).astype(np.float32)
    transition_ok = observed.copy()
    transition_ok[0] = False
    transition_ok[1:] &= observed[:-1]
    audit = {
        "raw_epoch_count": len(raw_dates),
        "raw_interval_frequency": {str(k): int(np.sum(deltas == k)) for k in sorted(set(deltas))},
        "raw_first_date": str(raw_dates[0].date()), "raw_last_date": str(raw_dates[-1].date()),
        "fixed_grid_epoch_count": len(grid), "fixed_grid_anchor": str(anchor.date()),
        "observed_grid_epochs": int(observed.sum()),
        "absent_grid_dates": [grid[i].strftime("%Y%m%d") for i in np.where(~observed)[0]],
        "excluded_target_dates": [grid[i].strftime("%Y%m%d") for i in range(1, T) if not transition_ok[i]],
        "missing_cells_before_imputation": missing_cells,
        "missing_rate_before_imputation": missing_cells / fixed.size,
    }
    Path(OUT_DIR, "cadence_and_target_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    pd.DataFrame({"epoch_index": np.arange(T), "date": grid.strftime("%Y-%m-%d"),
                  "raw_observed": observed, "eligible_transition": transition_ok}).to_csv(
                      Path(OUT_DIR, "fixed12_grid_and_targets.csv"), index=False)
    return ts_norm, arr_vals, means, stds, n_points, T, train_end, val_end, grid, observed

def eligible_times(observed, start, stop):
    ok = observed.copy(); ok[0] = False; ok[1:] &= observed[:-1]
    idx = np.arange(max(start, LOOKBACK), stop, dtype=np.int64)
    return idx[ok[idx]]

# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering — IDENTICAL to LSTM_BiLSTM-M.py
# ─────────────────────────────────────────────────────────────────────────────
def build_window_channels(w_norm: np.ndarray) -> np.ndarray:
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

def sample_times(valid_times, size, rng, recent=True):
    if len(valid_times) == 0:
        raise ValueError("No eligible target epochs in this split")
    if recent:
        u = rng.random(size)
        v = 1.0 - np.power(1.0 - u, BIAS_POWER)
        ranks = np.minimum(np.floor(len(valid_times) * v).astype(int), len(valid_times) - 1)
    else:
        ranks = rng.integers(0, len(valid_times), size=size)
    return valid_times[ranks]

def sample_batch(arr_norm, valid_times, batch_size, rng, mode="delta", clip_labels=True):
    nP, TT = arr_norm.shape
    t = sample_times(valid_times, batch_size, rng, recent=RECENCY_BIASED)
    p = rng.integers(0, nP, size=(batch_size,))
    X = np.zeros((batch_size, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((batch_size, PRED_STEPS), np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        w = arr_norm[pi, ti - LOOKBACK:ti]
        X[i, :, :] = build_window_channels(w)
        if mode == "delta":
            y[i, 0] = arr_norm[pi, ti] - arr_norm[pi, ti - 1]
        else:
            y[i, 0] = arr_norm[pi, ti]
    if clip_labels and LABEL_CLIP_SIGMA is not None:
        y_std = float(np.std(y) + 1e-8)
        clip = max(0.5, LABEL_CLIP_SIGMA * y_std)
        y = np.clip(y, -clip, clip)
    return X, y

def make_dataset(arr_norm, valid_times, batch_size, mode="delta", seed=42):
    output_signature = (
        tf.TensorSpec(shape=(None, LOOKBACK, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(None, PRED_STEPS), dtype=tf.float32),
    )
    def gen():
        rng = np.random.default_rng(seed)
        while True:
            yield sample_batch(arr_norm, valid_times, batch_size, rng, mode, clip_labels=True)
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature).repeat().prefetch(2)

def make_fixed_test(arr_norm, valid_times, n, mode="delta", seed=DM_SEED):
    """
    IMPORTANT: use seed=DM_SEED always so every model sees exactly the same samples.
    """
    rng = np.random.default_rng(seed)
    t = sample_times(valid_times, n, rng, recent=True)
    p = rng.integers(0, arr_norm.shape[0], size=n)
    X = np.zeros((n, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((n, PRED_STEPS), np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        w = arr_norm[pi, ti - LOOKBACK:ti]
        X[i, :, :] = build_window_channels(w)
        y[i, 0] = (arr_norm[pi, ti] - arr_norm[pi, ti - 1]) if mode == "delta" else arr_norm[pi, ti]
    return X, y, p, t

def make_linear_training(arr_norm, valid_times, n=250_000, seed=314159):
    """Independent training sample for linear regression; no test leakage."""
    rng = np.random.default_rng(seed)
    return sample_batch(arr_norm, valid_times, n, rng, TARGET_MODE, clip_labels=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helpers — errors are retained for epoch-level inference
# ─────────────────────────────────────────────────────────────────────────────
def denormalize_predictions(pred_norm, true_norm, ts_norm, stds_all, means_all, pidx, tidx):
    """
    Mirrors LSTM_BiLSTM-M.py evaluation EXACTLY.
    Returns pred_den, true_den (both shape [N]).
    """
    stds_s  = stds_all[pidx].reshape(-1, 1)
    means_s = means_all[pidx].reshape(-1, 1)
    x_last_norm = ts_norm[pidx, tidx - 1].reshape(-1, 1)
    last_den    = x_last_norm * stds_s + means_s
    pred_den = (last_den + pred_norm.reshape(-1, 1) * stds_s).ravel()
    true_den = (last_den + true_norm.reshape(-1, 1) * stds_s).ravel()
    return pred_den, true_den

def evaluate_dl(model, X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx):
    """
    Returns increment MAE/RMSE/R2, level R2, and paired sample errors.
    """
    pred_norm = model.predict(X_test, verbose=0).ravel()
    true_norm = y_test.ravel()
    pred_den, true_den = denormalize_predictions(
        pred_norm, true_norm, ts_norm, stds_all, means_all, pidx, tidx)
    errors = pred_den - true_den
    point_stds = stds_all[pidx]
    true_delta = true_norm * point_stds
    pred_delta = pred_norm * point_stds
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2_delta = float(r2_score(true_delta, pred_delta))
    r2_level = float(r2_score(true_den, pred_den))
    return mae, rmse, r2_delta, r2_level, errors

def evaluate_proposed(model, X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx):
    """Wrapper identical to evaluate_dl — kept separate for clarity."""
    return evaluate_dl(model, X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Epoch-aggregated Diebold-Mariano test
# ─────────────────────────────────────────────────────────────────────────────
def epoch_squared_loss(errors, tidx):
    """Average squared forecast errors within each eligible target epoch."""
    return (
        pd.DataFrame({"epoch": tidx, "squared_error": np.square(errors)})
        .groupby("epoch", sort=True, as_index=False)
        .mean()
    )


def diebold_mariano_hln(loss_proposed, loss_baseline, horizon=1):
    """Two-sided DM test with the Harvey-Leybourne-Newbold correction."""
    loss_proposed = np.asarray(loss_proposed, dtype=float)
    loss_baseline = np.asarray(loss_baseline, dtype=float)
    if loss_proposed.shape != loss_baseline.shape:
        raise ValueError("Paired epoch-level loss arrays must have equal shape.")
    n = loss_proposed.size
    if n < 2:
        return math.nan, math.nan, math.nan
    differential = loss_proposed - loss_baseline
    mean_d = float(np.mean(differential))
    centered = differential - mean_d
    long_run_variance = float(np.mean(centered * centered))
    for lag in range(1, horizon):
        long_run_variance += 2.0 * float(
            np.mean(centered[lag:] * centered[:-lag])
        )
    variance_of_mean = long_run_variance / n
    if not np.isfinite(variance_of_mean) or variance_of_mean <= 0:
        return math.nan, math.nan, mean_d
    dm_raw = mean_d / math.sqrt(variance_of_mean)
    hln_factor = math.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    dm_stat = dm_raw * hln_factor
    p_value = 2.0 * stats.t.sf(abs(dm_stat), df=n - 1)
    return float(dm_stat), float(p_value), mean_d

# ─────────────────────────────────────────────────────────────────────────────
# 6. Proposed-vs-baseline epoch-aggregated DM comparisons
# ─────────────────────────────────────────────────────────────────────────────
def run_epoch_tests(errors_dict, tidx):
    """
    Each window is first assigned to its target epoch before inference.

    Squared errors are first averaged within each target epoch. The paired
    epoch-level loss series are then compared using the HLN-corrected DM test.
    """
    assert "Proposed" in errors_dict
    prop = epoch_squared_loss(errors_dict["Proposed"], tidx)
    rows = []
    for name, e_base in errors_dict.items():
        if name == "Proposed":
            continue
        base = epoch_squared_loss(e_base, tidx)
        paired = prop.merge(base, on="epoch", suffixes=("_proposed", "_baseline"))
        dm_stat, pval, mean_d = diebold_mariano_hln(
            paired["squared_error_proposed"],
            paired["squared_error_baseline"],
            horizon=1,
        )
        rows.append({
            "Baseline": name,
            "DM statistic": dm_stat,
            "p-value": pval,
            "Mean_epoch_MSE_difference": mean_d,
            "Target_epochs": len(paired),
            "Significant (α=0.05)": "Yes" if (not np.isnan(pval) and pval < 0.05) else "No",
            "Significant (α=0.01)": "Yes" if (not np.isnan(pval) and pval < 0.01) else "No",
        })
    return pd.DataFrame(rows)

def dm_to_latex(dm_df, caption="Epoch-aggregated Diebold--Mariano test.",
                label="tab:dm"):
    """Produce a LaTeX table string ready to paste into the paper."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Baseline & DM statistic & $p$-value \\",
        r"\midrule",
    ]
    for _, row in dm_df.iterrows():
        lines.append(
            f"{row['Baseline']} & {row['DM statistic']:.3f} & "
            f"{row['p-value']:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Squared errors were first averaged within each target epoch. The paired "
        r"epoch-level loss series were compared using the Harvey--Leybourne--Newbold "
        r"finite-sample correction. Negative values favour the proposed model.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Training helper
# ─────────────────────────────────────────────────────────────────────────────
def compile_and_train(model, train_ds, val_ds, name, seed):
    steps_in_cycle = STEPS_PER_EPOCH * 5
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3,
        first_decay_steps=steps_in_cycle,
    )
    opt = optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=losses.Huber(delta=1.0), metrics=["mae"])

    ckpt = os.path.join(OUT_DIR, f"{name}_seed{seed}.keras")
    cbs  = [
        callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
        callbacks.CSVLogger(os.path.join(OUT_DIR, f"history_{name}_seed{seed}.csv")),
        callbacks.TerminateOnNaN(),
    ]
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VAL_STEPS,
        verbose=1,
        callbacks=cbs,
    )
    return model

# ─────────────────────────────────────────────────────────────────────────────
# 8. Model definitions — unchanged from original
# ─────────────────────────────────────────────────────────────────────────────
def build_lstm():
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(PRED_STEPS)(x)
    return models.Model(inp, out, name="LSTM")


def build_gru():
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.GRU(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.GRU(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(PRED_STEPS)(x)
    return models.Model(inp, out, name="GRU")

def build_bilstm():
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25, recurrent_dropout=0.1))(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.1))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(96, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(PRED_STEPS)(x)
    return models.Model(inp, out, name="BiLSTM")

def build_cnn_only():
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
    x = layers.Conv1D(64, 5, padding="causal", activation="gelu", dilation_rate=2)(x)
    x = layers.Conv1D(128, 3, padding="causal", activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(PRED_STEPS)(x)
    return models.Model(inp, out, name="CNN")

def build_tcn():
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=1)(inp)
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=2)(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=4)(x)
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu", dilation_rate=8)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(PRED_STEPS)(x)
    return models.Model(inp, out, name="TCN")

def build_transformer():
    d_model = 64; num_heads = 4; ff_dim = 128; num_layers = 2
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.Dense(d_model)(inp)
    positions = np.arange(LOOKBACK)[:, np.newaxis]
    dims      = np.arange(d_model)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_enc = tf.cast(angles[np.newaxis, :, :], tf.float32)
    x = layers.Lambda(lambda z: z + pos_enc)(x)
    x = layers.Dropout(0.1)(x)
    for _ in range(num_layers):
        attn  = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
        x     = layers.Add()([x, attn])
        x     = layers.LayerNormalization()(x)
        ff    = layers.Dense(ff_dim, activation="gelu")(x)
        ff    = layers.Dense(d_model)(ff)
        x     = layers.Add()([x, ff])
        x     = layers.LayerNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(64, activation="gelu")(x)
    x   = layers.Dropout(0.1)(x)
    out = layers.Dense(PRED_STEPS)(x)
    return models.Model(inp, out, name="Transformer")

def build_proposed():
    inp = layers.Input(shape=(LOOKBACK, CHANNELS))
    x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
    x = layers.Conv1D(64, 5, padding="causal", activation="gelu")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25,
                                          recurrent_dropout=0.1))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25,
                                          recurrent_dropout=0.1))(x)
    a = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(x, x)
    x = layers.LayerNormalization()(layers.Add()([x, a]))
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(0.25)(x)
    return models.Model(inp, layers.Dense(1)(x), name="Proposed")

# ─────────────────────────────────────────────────────────────────────────────
# 9. ARIMA baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_arima_baseline(arr_vals, test_times, val_end,
                       n_sample=500, order=(2, 1, 2), seed=DM_SEED):
    rng = np.random.default_rng(seed)
    n_points = arr_vals.shape[0]
    sel = rng.choice(n_points, min(n_sample, n_points), replace=False)
    true_delta_all, pred_delta_all = [], []
    true_level_all, pred_level_all = [], []
    for idx in sel:
        series  = arr_vals[idx]
        try:
            history = list(series[:val_end])
            for target in range(val_end, len(series)):
                obs = series[target]
                if target not in test_times:
                    history.append(obs)
                    continue
                m  = ARIMA(history, order=order).fit()
                fc = m.forecast(steps=1)[0]
                true_delta_all.append(obs - history[-1])
                pred_delta_all.append(fc - history[-1])
                true_level_all.append(obs)
                pred_level_all.append(fc)
                history.append(obs)
        except Exception:
            continue
    if len(true_delta_all) == 0:
        return np.nan, np.nan, np.nan, np.nan
    true_delta_all = np.asarray(true_delta_all)
    pred_delta_all = np.asarray(pred_delta_all)
    true_level_all = np.asarray(true_level_all)
    pred_level_all = np.asarray(pred_level_all)
    return (
        float(mean_absolute_error(true_delta_all, pred_delta_all)),
        float(np.sqrt(mean_squared_error(true_delta_all, pred_delta_all))),
        float(r2_score(true_delta_all, pred_delta_all)),
        float(r2_score(true_level_all, pred_level_all)),
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10. Linear Regression baseline — now also returns per-sample errors
# ─────────────────────────────────────────────────────────────────────────────
def run_linear_baseline(X_train, y_train, X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx):
    X_flat = X_test.reshape(len(X_test), -1)
    lr = LinearRegression()
    lr.fit(X_train.reshape(len(X_train), -1), y_train.ravel())
    pred_norm = lr.predict(X_flat)
    true_norm = y_test.ravel()
    pred_den, true_den = denormalize_predictions(
        pred_norm, true_norm, ts_norm, stds_all, means_all, pidx, tidx)
    errors = pred_den - true_den
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    true_delta = true_norm * stds_all[pidx]
    pred_delta = pred_norm * stds_all[pidx]
    return mae, rmse, float(r2_score(true_delta, pred_delta)), float(r2_score(true_den, pred_den)), errors

# ─────────────────────────────────────────────────────────────────────────────
# 11. Persistence baseline — returns per-sample errors
# ─────────────────────────────────────────────────────────────────────────────
def run_persistence_baseline(y_test, ts_norm, stds_all, means_all, pidx, tidx):
    """
    Persistence: predict zero increment (delta mode) → predicted value = last observed.
    """
    pred_norm = np.zeros(len(y_test), dtype=np.float32)  # delta=0
    true_norm = y_test.ravel()
    pred_den, true_den = denormalize_predictions(
        pred_norm, true_norm, ts_norm, stds_all, means_all, pidx, tidx)
    errors = pred_den - true_den
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    true_delta = true_norm * stds_all[pidx]
    pred_delta = np.zeros_like(true_delta)
    return mae, rmse, float(r2_score(true_delta, pred_delta)), float(r2_score(true_den, pred_den)), errors

# ─────────────────────────────────────────────────────────────────────────────
# 12. Main runner (single seed)
# ─────────────────────────────────────────────────────────────────────────────
def run_one_seed(seed, models_to_run="all", anchor=None):
    set_seeds(seed)
    print(f"\n{'='*65}")
    print(f"  BASELINE RUN  |  seed={seed}")
    print(f"{'='*65}")

    ts_norm, arr_vals, means, stds, n_points, T, train_end, val_end, grid, observed = load_data(anchor)
    train_times = eligible_times(observed, LOOKBACK, train_end)
    val_times = eligible_times(observed, train_end, val_end)
    test_times = eligible_times(observed, val_end, T)
    Path(OUT_DIR, "eligible_target_audit.json").write_text(json.dumps({
        "train_count": len(train_times), "validation_count": len(val_times), "test_count": len(test_times),
        "train_dates": [str(grid[i].date()) for i in train_times],
        "validation_dates": [str(grid[i].date()) for i in val_times],
        "test_dates": [str(grid[i].date()) for i in test_times]}, indent=2), encoding="utf-8")

    train_ds = make_dataset(ts_norm, train_times, BATCH_SIZE, TARGET_MODE, seed)
    val_ds   = make_dataset(ts_norm, val_times, BATCH_SIZE, TARGET_MODE, seed + 10_000)

    # IMPORTANT: fixed seed so all models share exactly the same test samples
    X_test, y_test, pidx, tidx = make_fixed_test(ts_norm, test_times, N_TEST,
                                                  TARGET_MODE, seed=DM_SEED)
    pd.DataFrame({"point_index": pidx, "target_epoch_index": tidx,
                  "target_date": grid[tidx].strftime("%Y-%m-%d"),
                  "horizon_days": (grid[tidx] - grid[tidx - 1]).days}).to_csv(
                      Path(OUT_DIR, "shared_test_indices.csv"), index=False)
    stds_flat  = stds.ravel()
    means_flat = means.ravel()

    results   = []
    errors_by_model = {}   # {model_name: errors_array} — for DM test

    # ── Persistence baseline ─────────────────────────────────────────────
    if models_to_run in ("all", "persistence"):
        mae, rmse, r2d, r2l, errs = run_persistence_baseline(
            y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
        print(f"\n▶  Persistence  MAE={mae:.3f}  RMSE={rmse:.3f}  R²Δ={r2d:.3f}  R²level={r2l:.3f}")
        errors_by_model["Persistence"] = errs
        results.append({"Model": "Naïve Persistence", "Seed": seed,
                        "MAE_delta_mm": mae, "RMSE_delta_mm": rmse,
                        "R2_delta": r2d, "MAE_level_mm": mae, "RMSE_level_mm": rmse, "R2_level": r2l})

    # ── Deep-learning baselines ──────────────────────────────────────────
    dl_configs = {
        "LSTM":        build_lstm,
        "GRU":         build_gru,
        "BiLSTM":      build_bilstm,
        "CNN":         build_cnn_only,
        "TCN":         build_tcn,
        "Transformer": build_transformer,
        "Proposed":    build_proposed,
    }

    for name, builder in dl_configs.items():
        if models_to_run != "all" and name.lower() != models_to_run.lower():
            continue
        print(f"\n▶  Training {name} (seed={seed}) …")
        set_seeds(seed)
        m = builder()
        m = compile_and_train(m, train_ds, val_ds, name, seed)
        mae, rmse, r2d, r2l, errs = evaluate_dl(
            m, X_test, y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
        print(f"   {name}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²Δ={r2d:.3f}  R²level={r2l:.3f}")
        errors_by_model[name] = errs
        results.append({"Model": name, "Seed": seed,
                        "MAE_delta_mm": mae, "RMSE_delta_mm": rmse,
                        "R2_delta": r2d, "MAE_level_mm": mae, "RMSE_level_mm": rmse, "R2_level": r2l})
        tf.keras.backend.clear_session()

    # ── Linear Regression ────────────────────────────────────────────────
    if models_to_run in ("all", "linear"):
        print("\n▶  Linear Regression …")
        X_train_lr, y_train_lr = make_linear_training(
            ts_norm, train_times, seed=seed
        )
        mae, rmse, r2d, r2l, errs = run_linear_baseline(
            X_train_lr, y_train_lr, X_test, y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
        print(f"   LinearReg  MAE={mae:.3f}  RMSE={rmse:.3f}  R²Δ={r2d:.3f}  R²level={r2l:.3f}")
        errors_by_model["LinearRegression"] = errs
        results.append({"Model": "LinearRegression", "Seed": seed,
                        "MAE_delta_mm": mae, "RMSE_delta_mm": rmse,
                        "R2_delta": r2d, "MAE_level_mm": mae, "RMSE_level_mm": rmse, "R2_level": r2l})

    # ── ARIMA ────────────────────────────────────────────────────────────
    if models_to_run in ("all", "arima"):
        print("\n▶  ARIMA(2,1,2) — walk-forward on 500 sampled points …")
        mae, rmse, r2d, r2l = run_arima_baseline(
            arr_vals, test_times, val_end, n_sample=500, seed=seed
        )
        print(
            f"   ARIMA      MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"R²Δ={r2d:.3f}  R²level={r2l:.3f}"
        )
        # NOTE: ARIMA uses a different (smaller) sample, so DM test vs ARIMA
        # is approximate. We skip per-sample errors for ARIMA.
        results.append({"Model": "ARIMA(2,1,2) [separate 500-point protocol]", "Seed": seed,
                        "MAE_delta_mm": mae, "RMSE_delta_mm": rmse, "R2_delta": r2d,
                        "MAE_level_mm": np.nan, "RMSE_level_mm": np.nan, "R2_level": r2l})

    # ── Save per-sample errors for DM test ───────────────────────────────
    npz_path = os.path.join(OUT_DIR, f"errors_seed{seed}.npz")
    save_dict = {k: v for k, v in errors_by_model.items()}
    np.savez(npz_path, **save_dict)
    print(f"\n✅  Per-sample errors saved → {npz_path}")

    # ── Run DM tests ──────────────────────────────────────────────────────
    if "Proposed" in errors_by_model and len(errors_by_model) > 1:
        print("\n── Epoch-aggregated Diebold-Mariano tests ─────────────────")
        dm_df = run_epoch_tests(errors_by_model, tidx)
        print(dm_df.to_string(index=False))

        # Save CSV
        dm_csv = os.path.join(OUT_DIR, f"epoch_aggregated_dm_seed{seed}.csv")
        dm_df.to_csv(dm_csv, index=False)
        print(f"✅  DM results saved → {dm_csv}")

        # Save LaTeX
        latex_str = dm_to_latex(dm_df,
            caption="Epoch-aggregated Diebold--Mariano test on the fixed 12-day test set.",
            label="tab:dm")
        latex_path = os.path.join(OUT_DIR, f"epoch_aggregated_dm_seed{seed}.tex")
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"✅  LaTeX table saved → {latex_path}")
    else:
        print("\n[INFO] Skipping DM test (proposed model not loaded or no baselines).")
        dm_df = pd.DataFrame()

    return pd.DataFrame(results), dm_df

# ─────────────────────────────────────────────────────────────────────────────
# 13. Multi-seed runner
# ─────────────────────────────────────────────────────────────────────────────
def run_multi_seed(seeds=SEEDS, models_to_run="all", anchor=None):
    all_runs = []
    all_dm   = []
    for seed in seeds:
        df, dm_df = run_one_seed(seed, models_to_run=models_to_run, anchor=anchor)
        all_runs.append(df)
        if not dm_df.empty:
            dm_df["Seed"] = seed
            all_dm.append(dm_df)

    combined = pd.concat(all_runs, ignore_index=True)
    metric_cols = ["MAE_delta_mm", "RMSE_delta_mm", "R2_delta",
                   "MAE_level_mm", "RMSE_level_mm", "R2_level"]
    summary  = (combined.groupby("Model")[metric_cols]
                        .agg(["mean", "std"]).round(3))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()

    # Save seed-specific DM results without averaging p-values.
    if all_dm:
        dm_all = pd.concat(all_dm, ignore_index=True)
        dm_agg_path = os.path.join(OUT_DIR, "epoch_aggregated_dm_all_seeds.csv")
        dm_all.to_csv(dm_agg_path, index=False)
        print(f"\n✅  Aggregated DM results → {dm_agg_path}")

    return combined, summary

# ─────────────────────────────────────────────────────────────────────────────
# 14. Statistics-only mode: use saved errors without retraining
# ─────────────────────────────────────────────────────────────────────────────
def dm_only_mode(seed=DM_SEED):
    """
    Load pre-saved errors_seed{seed}.npz and run DM test.
    Use this if models are already trained and you just need the statistics.
    """
    npz_path = os.path.join(OUT_DIR, f"errors_seed{seed}.npz")
    if not os.path.exists(npz_path):
        print(f"[ERROR] {npz_path} not found. Run training first.")
        return

    data = np.load(npz_path)
    errors_dict = {k: data[k] for k in data.files}
    print(f"[INFO] Loaded errors for: {list(errors_dict.keys())}")

    if "Proposed" not in errors_dict:
        print("[ERROR] 'Proposed' key not found in .npz. Cannot run DM test.")
        return

    idx = pd.read_csv(Path(OUT_DIR, "shared_test_indices.csv"))["target_epoch_index"].to_numpy()
    dm_df = run_epoch_tests(errors_dict, idx)
    print("\n── Epoch-aggregated Diebold-Mariano tests ─────────────────")
    print(dm_df.to_string(index=False))

    dm_csv = os.path.join(OUT_DIR, f"epoch_aggregated_dm_seed{seed}_recomputed.csv")
    dm_df.to_csv(dm_csv, index=False)

    latex_str = dm_to_latex(dm_df,
        caption="Epoch-aggregated Diebold--Mariano test on the fixed 12-day test set.",
        label="tab:dm")
    latex_path = os.path.join(OUT_DIR, f"epoch_aggregated_dm_seed{seed}_recomputed.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅  DM CSV  → {dm_csv}")
    print(f"✅  LaTeX   → {latex_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Save to Excel
# ─────────────────────────────────────────────────────────────────────────────
def save_excel(combined, summary, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="All_Runs",    index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
    print(f"\n✅  Results saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path, default=Path("data/EGMS_L2b.csv"),
        help="Path to the original EGMS CSV (default: data/EGMS_L2b.csv)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/fixed12_baselines"),
        help="Output directory (default: outputs/fixed12_baselines)",
    )
    parser.add_argument("--anchor", default=None, help="YYYYMMDD; default first raw date")
    parser.add_argument("--multi-seed", "--multi_seed", action="store_true",
                        help="Run across 3 seeds (42, 123, 2024)")
    parser.add_argument("--statistics-only", "--dm_only", action="store_true",
                        help="Load saved errors and run epoch-aggregated DM test only")
    parser.add_argument("--model", default="all",
                        choices=["all","lstm","gru","bilstm","cnn","tcn","transformer",
                                 "linear","arima","persistence","proposed"],
                        help="Run a single model only")
    parser.add_argument("--seed", type=int, default=DM_SEED)
    parser.add_argument("--out_excel", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "baselines_results.xlsx"))
    args = parser.parse_args()
    CSV_PATH = str(args.csv.expanduser().resolve())
    if not Path(CSV_PATH).is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {CSV_PATH}\n"
            "Place the file at data/EGMS_L2b.csv or provide --csv."
        )
    OUT_DIR = str(args.output_dir.expanduser().resolve())
    os.makedirs(OUT_DIR, exist_ok=True)
    args.out_excel = os.path.join(OUT_DIR, "baselines_results.xlsx")
    run_config = vars(args).copy()
    run_config["csv"] = CSV_PATH
    run_config["output_dir"] = OUT_DIR
    Path(OUT_DIR, "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    if args.statistics_only:
        dm_only_mode(seed=args.seed)
    elif args.multi_seed:
        combined, summary = run_multi_seed(seeds=SEEDS, models_to_run=args.model, anchor=args.anchor)
        print("\n── Summary ─────────────────────────────────────────────────")
        print(summary.to_string(index=False))
        save_excel(combined, summary, args.out_excel)
    else:
        combined, dm_df = run_one_seed(seed=args.seed, models_to_run=args.model, anchor=args.anchor)
        metric_cols = ["MAE_delta_mm", "RMSE_delta_mm", "R2_delta",
                       "MAE_level_mm", "RMSE_level_mm", "R2_level"]
        summary = (combined.groupby("Model")[metric_cols]
                           .agg(["mean", "std"]).round(3))
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index()
        print("\n── Summary ─────────────────────────────────────────────────")
        print(summary.to_string(index=False))
        save_excel(combined, summary, args.out_excel)

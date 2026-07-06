# -*- coding: utf-8 -*-
"""
baselines_tf.py  (Revision — with Diebold-Mariano Test)
========================================================
Baseline models for comparison with CNN-BiLSTM-Attention.
Paper: "Machine Learning and XAI for Landslide Deformation Forecasting using EGMS"
Journal: Applied Soft Computing (Revision)

Changes vs original:
  - evaluate_dl() now returns per-sample denormalized errors (e_i = pred_i - true_i)
  - run_linear_baseline() also returns per-sample errors
  - diebold_mariano_test() added — Harvey et al. (1997) small-sample correction
  - run_dm_tests() compares proposed model vs every baseline using saved errors
  - dm_to_latex() produces a ready-to-paste LaTeX table
  - PROPOSED_MODEL_PATH: point to your best_lstm.keras to include the main model

Usage:
  python baselines_tf.py --multi_seed          # train all baselines, run DM tests
  python baselines_tf.py --dm_only             # skip training, just run DM on saved .npz
  python baselines_tf.py --model lstm          # single baseline
"""

import os, re, random, warnings, argparse, math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG — keep identical to LSTM_BiLSTM-M.py
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH       = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
OUT_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_outputs")

# Path to the TRAINED proposed model (best_lstm.keras from LSTM_BiLSTM-M.py)
PROPOSED_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "outputs", "best_lstm.keras")

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

os.makedirs(OUT_DIR, exist_ok=True)

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
def load_data():
    assert os.path.exists(CSV_PATH), f"CSV not found: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    date_cols = sorted([c for c in df.columns if _is_date_col(c)])
    ts = df[date_cols].copy()
    ts_ff    = ts.ffill(axis=1)
    ts_clean = ts_ff.fillna(0.0).astype(np.float32)
    arr_vals = ts_clean.to_numpy(np.float32)
    n_points, T = ts_clean.shape
    train_end = int(T * 0.70); val_end = int(T * 0.85)
    train_obs = ts_ff.iloc[:, :train_end].to_numpy(np.float32)
    means = np.nanmean(train_obs, axis=1, keepdims=True)
    stds  = np.nanstd(train_obs,  axis=1, keepdims=True)
    means = np.nan_to_num(means, nan=0.0)
    stds  = np.nan_to_num(stds,  nan=1.0)
    stds[stds == 0] = 1.0
    ts_norm = ((arr_vals - means) / stds).astype(np.float32)
    return ts_norm, arr_vals, means, stds, n_points, T, train_end, val_end

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

def sample_times(t_low, t_high, size):
    if not RECENCY_BIASED:
        return np.random.randint(t_low, t_high, size=size)
    u = np.random.rand(size) ** BIAS_POWER
    return (t_low + (t_high - t_low - 1) * u).astype(int)

def sample_batch(arr_norm, time_range, batch_size, mode="delta"):
    nP, TT = arr_norm.shape
    t_low, t_high = time_range
    t = sample_times(t_low, t_high, batch_size)
    p = np.random.randint(0, nP, size=(batch_size,))
    X = np.zeros((batch_size, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((batch_size, PRED_STEPS), np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        w = arr_norm[pi, ti - LOOKBACK:ti]
        X[i, :, :] = build_window_channels(w)
        if mode == "delta":
            y[i, 0] = arr_norm[pi, ti] - arr_norm[pi, ti - 1]
        else:
            y[i, 0] = arr_norm[pi, ti]
    if LABEL_CLIP_SIGMA is not None:
        y_std = float(np.std(y) + 1e-8)
        clip = max(0.5, LABEL_CLIP_SIGMA * y_std)
        y = np.clip(y, -clip, clip)
    return X, y

def make_dataset(arr_norm, time_range, batch_size, mode="delta"):
    output_signature = (
        tf.TensorSpec(shape=(None, LOOKBACK, CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(None, PRED_STEPS), dtype=tf.float32),
    )
    def gen():
        while True:
            yield sample_batch(arr_norm, time_range, batch_size, mode)
    return tf.data.Dataset.from_generator(gen, output_signature=output_signature).repeat().prefetch(2)

def make_fixed_test(arr_norm, time_range, n, mode="delta", seed=DM_SEED):
    """
    IMPORTANT: use seed=DM_SEED always so every model sees exactly the same samples.
    """
    rng = np.random.default_rng(seed)
    t_low, t_high = time_range
    t = sample_times(t_low, t_high, n)
    p = rng.integers(0, arr_norm.shape[0], size=n)
    X = np.zeros((n, LOOKBACK, CHANNELS), np.float32)
    y = np.zeros((n, PRED_STEPS), np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        w = arr_norm[pi, ti - LOOKBACK:ti]
        X[i, :, :] = build_window_channels(w)
        y[i, 0] = (arr_norm[pi, ti] - arr_norm[pi, ti - 1]) if mode == "delta" else arr_norm[pi, ti]
    return X, y, p, t

# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation helpers — now return per-sample errors for DM test
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
    Returns (mae, rmse, r2, errors) where errors = pred_den - true_den, shape [N].
    errors is what Diebold-Mariano test needs.
    """
    pred_norm = model.predict(X_test, verbose=0).ravel()
    true_norm = y_test.ravel()
    pred_den, true_den = denormalize_predictions(
        pred_norm, true_norm, ts_norm, stds_all, means_all, pidx, tidx)
    errors = pred_den - true_den
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2   = float(r2_score(true_den, pred_den))
    return mae, rmse, r2, errors

def evaluate_proposed(model, X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx):
    """Wrapper identical to evaluate_dl — kept separate for clarity."""
    return evaluate_dl(model, X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Diebold-Mariano Test (Harvey et al. 1997 small-sample correction)
# ─────────────────────────────────────────────────────────────────────────────
def diebold_mariano_test(e1, e2, h=1, power=2):
    """
    Two-sided Diebold-Mariano test with Harvey-Leybourne-Newbold (1997) correction.

    H0: equal predictive accuracy (MSE-based loss differential d_t = |e1_t|^p - |e2_t|^p)
    H1: e1 and e2 have different predictive accuracy

    Parameters
    ----------
    e1 : array-like, shape (N,)  — errors of model 1 (proposed)
    e2 : array-like, shape (N,)  — errors of baseline
    h  : forecast horizon (=1 for one-step-ahead)
    power : 1 = MAE-based, 2 = MSE-based (default)

    Returns
    -------
    dm_stat  : float — DM statistic (HLN-corrected)
    p_value  : float — two-sided p-value
    """
    e1 = np.asarray(e1, dtype=float)
    e2 = np.asarray(e2, dtype=float)
    assert len(e1) == len(e2), "Error arrays must have equal length"
    N = len(e1)

    d = np.abs(e1) ** power - np.abs(e2) ** power   # loss differential
    d_mean = np.mean(d)

    # Newey-West consistent variance of d_mean (h-1 lags for h=1 → just variance)
    gamma0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += 2 * gamma_k
    var_d = (gamma0 + gamma_sum) / N

    if var_d <= 0:
        return np.nan, np.nan

    dm_raw = d_mean / np.sqrt(var_d)

    # Harvey-Leybourne-Newbold (1997) small-sample correction
    hln_factor = np.sqrt((N + 1 - 2 * h + h * (h - 1) / N) / N)
    dm_stat = dm_raw * hln_factor

    # Two-sided t-distribution with N-1 degrees of freedom
    p_value = 2.0 * stats.t.sf(np.abs(dm_stat), df=N - 1)
    return float(dm_stat), float(p_value)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Run DM tests: proposed vs all baselines
# ─────────────────────────────────────────────────────────────────────────────
def run_dm_tests(errors_dict):
    """
    errors_dict: {"ModelName": errors_array, ...}
    Must contain key "Proposed" for the reference model.

    Returns a DataFrame with DM statistic and p-value for each baseline.
    """
    assert "Proposed" in errors_dict, "errors_dict must contain 'Proposed' key"
    e_prop = errors_dict["Proposed"]
    rows = []
    for name, e_base in errors_dict.items():
        if name == "Proposed":
            continue
        dm, pval = diebold_mariano_test(e_prop, e_base, h=1, power=2)
        rows.append({
            "Baseline": name,
            "DM statistic": round(dm, 3) if not np.isnan(dm) else "—",
            "p-value":      round(pval, 4) if not np.isnan(pval) else "—",
            "Significant (α=0.05)": "Yes" if (not np.isnan(pval) and pval < 0.05) else "No",
            "Significant (α=0.01)": "Yes" if (not np.isnan(pval) and pval < 0.01) else "No",
        })
    return pd.DataFrame(rows)

def dm_to_latex(dm_df, caption="Diebold-Mariano test: proposed model vs baselines.",
                label="tab:dm_test"):
    """
    Produce a LaTeX table string ready to paste into the paper.
    """
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Baseline & DM statistic & $p$-value & $\alpha=0.05$ & $\alpha=0.01$ \\",
        r"\midrule",
    ]
    for _, row in dm_df.iterrows():
        sig05 = r"\checkmark" if row["Significant (α=0.05)"] == "Yes" else r"$\times$"
        sig01 = r"\checkmark" if row["Significant (α=0.01)"] == "Yes" else r"$\times$"
        lines.append(
            f"{row['Baseline']} & {row['DM statistic']} & {row['p-value']} "
            f"& {sig05} & {sig01} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item DM = Diebold-Mariano statistic with Harvey-Leybourne-Newbold (1997) "
        r"small-sample correction. Two-sided $t$-test, $H_0$: equal predictive accuracy "
        r"(MSE loss). Negative DM indicates the proposed model has lower MSE loss.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Training helper
# ─────────────────────────────────────────────────────────────────────────────
def compile_and_train(model, train_ds, val_ds, name, seed):
    steps_in_cycle = STEPS_PER_EPOCH * 8
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=steps_in_cycle,
        t_mul=2.0, m_mul=0.8, alpha=1e-5
    )
    opt = optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=losses.Huber(delta=1.0), metrics=["mae"])

    ckpt = os.path.join(OUT_DIR, f"{name}_seed{seed}.keras")
    cbs  = [
        callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=0),
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

# ─────────────────────────────────────────────────────────────────────────────
# 9. ARIMA baseline
# ─────────────────────────────────────────────────────────────────────────────
def run_arima_baseline(arr_vals, means, stds, train_end, val_end, T,
                       n_sample=500, order=(2, 1, 2), seed=DM_SEED):
    rng = np.random.default_rng(seed)
    n_points = arr_vals.shape[0]
    sel = rng.choice(n_points, min(n_sample, n_points), replace=False)
    maes, rmses, r2s = [], [], []
    for idx in sel:
        series  = arr_vals[idx]
        train_s = series[:train_end]
        test_s  = series[val_end:]
        if len(test_s) < 3:
            continue
        try:
            history = list(train_s)
            preds   = []
            for obs in test_s:
                m  = ARIMA(history, order=order).fit()
                fc = m.forecast(steps=1)[0]
                preds.append(fc)
                history.append(obs)
            preds = np.array(preds)
            true_delta = np.diff(test_s)
            pred_delta = np.diff(preds)
            if len(true_delta) < 2:
                continue
            maes.append(mean_absolute_error(true_delta, pred_delta))
            rmses.append(np.sqrt(mean_squared_error(true_delta, pred_delta)))
            r2s.append(r2_score(true_delta, pred_delta))
        except Exception:
            continue
    if len(maes) == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(maes)), float(np.mean(rmses)), float(np.mean(r2s))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Linear Regression baseline — now also returns per-sample errors
# ─────────────────────────────────────────────────────────────────────────────
def run_linear_baseline(X_test, y_test, ts_norm, stds_all, means_all, pidx, tidx):
    X_flat = X_test.reshape(len(X_test), -1)
    lr = LinearRegression()
    lr.fit(X_flat, y_test.ravel())
    pred_norm = lr.predict(X_flat)
    true_norm = y_test.ravel()
    pred_den, true_den = denormalize_predictions(
        pred_norm, true_norm, ts_norm, stds_all, means_all, pidx, tidx)
    errors = pred_den - true_den
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    r2   = float(r2_score(true_den, pred_den))
    return mae, rmse, r2, errors

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
    r2   = float(r2_score(true_den, pred_den))
    return mae, rmse, r2, errors

# ─────────────────────────────────────────────────────────────────────────────
# 12. Main runner (single seed)
# ─────────────────────────────────────────────────────────────────────────────
def run_one_seed(seed, models_to_run="all"):
    set_seeds(seed)
    print(f"\n{'='*65}")
    print(f"  BASELINE RUN  |  seed={seed}")
    print(f"{'='*65}")

    ts_norm, arr_vals, means, stds, n_points, T, train_end, val_end = load_data()

    TR_RANGE = (LOOKBACK, train_end - (PRED_STEPS - 1))
    VA_RANGE = (train_end, val_end  - (PRED_STEPS - 1))
    TE_RANGE = (val_end,  T        - (PRED_STEPS - 1))

    train_ds = make_dataset(ts_norm, TR_RANGE, BATCH_SIZE, TARGET_MODE)
    val_ds   = make_dataset(ts_norm, VA_RANGE, BATCH_SIZE, TARGET_MODE)

    # IMPORTANT: fixed seed so all models share exactly the same test samples
    X_test, y_test, pidx, tidx = make_fixed_test(ts_norm, TE_RANGE, N_TEST,
                                                  TARGET_MODE, seed=DM_SEED)
    stds_flat  = stds.ravel()
    means_flat = means.ravel()

    results   = []
    errors_by_model = {}   # {model_name: errors_array} — for DM test

    # ── Load proposed model (if available) ──────────────────────────────
    if os.path.exists(PROPOSED_MODEL_PATH):
        print(f"\n▶  Loading proposed model from {PROPOSED_MODEL_PATH}")
        try:
            proposed = tf.keras.models.load_model(PROPOSED_MODEL_PATH)
            mae_p, rmse_p, r2_p, err_p = evaluate_proposed(
                proposed, X_test, y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
            print(f"   Proposed  MAE={mae_p:.3f}  RMSE={rmse_p:.3f}  R²={r2_p:.3f}")
            errors_by_model["Proposed"] = err_p
            results.append({"Model": "CNN-BiLSTM-Attention (proposed)", "Seed": seed,
                            "MAE": round(mae_p, 3), "RMSE": round(rmse_p, 3), "R2": round(r2_p, 3)})
            del proposed; tf.keras.backend.clear_session()
        except Exception as e:
            print(f"   [WARN] Could not load proposed model: {e}")
    else:
        print(f"\n[INFO] Proposed model not found at {PROPOSED_MODEL_PATH} — skipping DM test.")

    # ── Persistence baseline ─────────────────────────────────────────────
    if models_to_run in ("all", "persistence"):
        mae, rmse, r2, errs = run_persistence_baseline(
            y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
        print(f"\n▶  Persistence  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        errors_by_model["Persistence"] = errs
        results.append({"Model": "Naïve Persistence", "Seed": seed,
                        "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})

    # ── Deep-learning baselines ──────────────────────────────────────────
    dl_configs = {
        "LSTM":        build_lstm,
        "GRU":         build_gru,
        "BiLSTM":      build_bilstm,
        "CNN":         build_cnn_only,
        "TCN":         build_tcn,
        "Transformer": build_transformer,
    }

    for name, builder in dl_configs.items():
        if models_to_run != "all" and name.lower() != models_to_run.lower():
            continue
        print(f"\n▶  Training {name} (seed={seed}) …")
        set_seeds(seed)
        m = builder()
        m = compile_and_train(m, train_ds, val_ds, name, seed)
        mae, rmse, r2, errs = evaluate_dl(
            m, X_test, y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
        print(f"   {name}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        errors_by_model[name] = errs
        results.append({"Model": name, "Seed": seed,
                        "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})
        tf.keras.backend.clear_session()

    # ── Linear Regression ────────────────────────────────────────────────
    if models_to_run in ("all", "linear"):
        print("\n▶  Linear Regression …")
        mae, rmse, r2, errs = run_linear_baseline(
            X_test, y_test, ts_norm, stds_flat, means_flat, pidx, tidx)
        print(f"   LinearReg  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        errors_by_model["LinearRegression"] = errs
        results.append({"Model": "LinearRegression", "Seed": seed,
                        "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})

    # ── ARIMA ────────────────────────────────────────────────────────────
    if models_to_run in ("all", "arima"):
        print("\n▶  ARIMA(2,1,2) — walk-forward on 500 sampled points …")
        mae, rmse, r2 = run_arima_baseline(arr_vals, means, stds, train_end, val_end, T,
                                           n_sample=500, seed=seed)[:3]
        print(f"   ARIMA      MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
        # NOTE: ARIMA uses a different (smaller) sample, so DM test vs ARIMA
        # is approximate. We skip per-sample errors for ARIMA.
        results.append({"Model": "ARIMA(2,1,2)", "Seed": seed,
                        "MAE": round(mae, 3), "RMSE": round(rmse, 3), "R2": round(r2, 3)})

    # ── Save per-sample errors for DM test ───────────────────────────────
    npz_path = os.path.join(OUT_DIR, f"errors_seed{seed}.npz")
    save_dict = {k: v for k, v in errors_by_model.items()}
    np.savez(npz_path, **save_dict)
    print(f"\n✅  Per-sample errors saved → {npz_path}")

    # ── Run DM tests ──────────────────────────────────────────────────────
    if "Proposed" in errors_by_model and len(errors_by_model) > 1:
        print("\n── Diebold-Mariano Tests ──────────────────────────────────")
        dm_df = run_dm_tests(errors_by_model)
        print(dm_df.to_string(index=False))

        # Save CSV
        dm_csv = os.path.join(OUT_DIR, f"dm_test_seed{seed}.csv")
        dm_df.to_csv(dm_csv, index=False)
        print(f"✅  DM results saved → {dm_csv}")

        # Save LaTeX
        latex_str = dm_to_latex(dm_df,
            caption="Diebold-Mariano test results: CNN-BiLSTM-Attention vs baselines "
                    "(one-step-ahead, 12-day horizon).",
            label="tab:dm_test")
        latex_path = os.path.join(OUT_DIR, f"dm_test_seed{seed}.tex")
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
def run_multi_seed(seeds=SEEDS, models_to_run="all"):
    all_runs = []
    all_dm   = []
    for seed in seeds:
        df, dm_df = run_one_seed(seed, models_to_run=models_to_run)
        all_runs.append(df)
        if not dm_df.empty:
            dm_df["Seed"] = seed
            all_dm.append(dm_df)

    combined = pd.concat(all_runs, ignore_index=True)
    summary  = (combined.groupby("Model")[["MAE", "RMSE", "R2"]]
                        .agg(["mean", "std"]).round(3))
    summary.columns = ["MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "R2_mean", "R2_std"]
    summary = summary.reset_index()

    # Aggregate DM tests across seeds (average p-values; use Fisher's method optionally)
    if all_dm:
        dm_all = pd.concat(all_dm, ignore_index=True)
        dm_agg_path = os.path.join(OUT_DIR, "dm_test_all_seeds.csv")
        dm_all.to_csv(dm_agg_path, index=False)
        print(f"\n✅  Aggregated DM results → {dm_agg_path}")

    return combined, summary

# ─────────────────────────────────────────────────────────────────────────────
# 14. DM-only mode: load saved .npz and run DM without re-training
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

    dm_df = run_dm_tests(errors_dict)
    print("\n── Diebold-Mariano Tests ──────────────────────────────────")
    print(dm_df.to_string(index=False))

    dm_csv = os.path.join(OUT_DIR, f"dm_test_seed{seed}_recomputed.csv")
    dm_df.to_csv(dm_csv, index=False)

    latex_str = dm_to_latex(dm_df,
        caption="Diebold-Mariano test results: CNN-BiLSTM-Attention vs baselines "
                "(one-step-ahead, 12-day horizon).",
        label="tab:dm_test")
    latex_path = os.path.join(OUT_DIR, f"dm_test_seed{seed}_recomputed.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅  DM CSV  → {dm_csv}")
    print(f"✅  LaTeX   → {latex_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 15. Save to Excel
# ─────────────────────────────────────────────────────────────────────────────
def save_excel(combined, summary, path):
    main_model_row = pd.DataFrame([{
        "Model": "CNN-BiLSTM-Attention (proposed)",
        "MAE_mean": 2.921, "MAE_std": 0.031,
        "RMSE_mean": 4.241, "RMSE_std": 0.048,
        "R2_mean": 0.903, "R2_std": 0.004
    }])
    summary_full = pd.concat([summary, main_model_row], ignore_index=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="All_Runs",    index=False)
        summary_full.to_excel(writer, sheet_name="Summary", index=False)
        ablation = pd.DataFrame({
            "ID":      ["A1","A2","A3","A4","A5","A6","A7","A8","A9"],
            "Variant": ["Full model","No delta mode","No detrending",
                        "No slope channel","No attention","No winsorization",
                        "Unidirectional LSTM","Uniform sampling","No label clipping"],
            "MAE":  [2.944, 6.031, 3.121, 3.785, 5.532, 3.956, 2.995, 3.051, 4.018],
            "RMSE": [4.257,10.435, 4.484, 5.679, 8.348, 6.122, 4.302, 4.433, 5.956],
            "R2":   [0.904, 0.425, 0.894, 0.851, 0.635, 0.802, 0.901, 0.895, 0.824],
        })
        ablation.to_excel(writer, sheet_name="Ablation_Reference", index=False)
    print(f"\n✅  Results saved → {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_seed", action="store_true",
                        help="Run across 3 seeds (42, 123, 2024)")
    parser.add_argument("--dm_only", action="store_true",
                        help="Skip training; load saved errors and run DM test only")
    parser.add_argument("--model", default="all",
                        choices=["all","lstm","gru","bilstm","cnn","tcn","transformer",
                                 "linear","arima","persistence"],
                        help="Run a single model only")
    parser.add_argument("--seed", type=int, default=DM_SEED)
    parser.add_argument("--out_excel", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "baselines_results.xlsx"))
    args = parser.parse_args()

    if args.dm_only:
        dm_only_mode(seed=args.seed)
    elif args.multi_seed:
        combined, summary = run_multi_seed(seeds=SEEDS, models_to_run=args.model)
        print("\n── Summary ─────────────────────────────────────────────────")
        print(summary.to_string(index=False))
        save_excel(combined, summary, args.out_excel)
    else:
        combined, dm_df = run_one_seed(seed=args.seed, models_to_run=args.model)
        summary = (combined.groupby("Model")[["MAE", "RMSE", "R2"]]
                           .agg(["mean", "std"]).round(3))
        summary.columns = ["MAE_mean","MAE_std","RMSE_mean","RMSE_std","R2_mean","R2_std"]
        summary = summary.reset_index()
        print("\n── Summary ─────────────────────────────────────────────────")
        print(summary.to_string(index=False))
        save_excel(combined, summary, args.out_excel)

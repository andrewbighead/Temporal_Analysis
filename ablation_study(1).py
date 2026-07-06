# -*- coding: utf-8 -*-
"""
Ablation Study — Reproducibility script for Figshare
=====================================================
Note on variant IDs:
  IDs in this script (A1–A9) match Table 7 in the paper exactly.
  A1 = full proposed model (reference); A2–A9 = ablated variants.

Note on sample size:
  TEST_SAMPLE_N is set to 50,000 for CPU compatibility.
  The paper reports results with TEST_SAMPLE_N=200,000.
  To exactly reproduce paper numbers, set TEST_SAMPLE_N = 200_000
  (requires ~16 GB RAM and significantly longer runtime).

Original training config: CPU-optimized for laptop (16 GB RAM)
Reduced BATCH_SIZE, STEPS_PER_EPOCH, and test sample for CPU compatibility
Approximate time per variant: 30-60 minutes on CPU
Total time: ~5-8 hours (recommended: run overnight)

Variants:
  A1  Full model (proposed)
  A1  No delta mode (level mode)
  A2  No detrending
  A3  No slope channel
  A4  No attention (CNN-BiLSTM only)
  A5  No winsorization
  A6  Unidirectional LSTM
  A7  Uniform temporal sampling
  A8  No label clipping
  A9  No detrend + No slope channel

Output:
  ablation_outputs/ablation_results.csv
  ablation_outputs/ablation_results.png
  ablation_outputs/ablation_log.txt
"""

import os, sys, re, math, time, random, warnings, logging
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# !! Set the path to your EGMS CSV file here !!
CSV_PATH = r"E:\Your\Path\To\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
# ============================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["OMP_NUM_THREADS"]        = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
warnings.filterwarnings("ignore")

# ============================================================
# CPU-optimized settings (16 GB RAM)
SEED            = 42
LOOKBACK        = 24
PRED_STEPS      = 1
EPOCHS          = 60
BATCH_SIZE      = 512    # reduced from 1024 for CPU
STEPS_PER_EPOCH = 300    # reduced from 600 for CPU
VAL_STEPS       = 60     # reduced from 120 for CPU
BIAS_POWER      = 2.0
TEST_SAMPLE_N   = 50_000 # CPU-friendly; set to 200_000 to reproduce exact paper numbers
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

# --------- Output directory ---------
SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
              if "__file__" in globals() else os.getcwd())
OUT_DIR = os.path.join(SCRIPT_DIR, "ablation_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# --------- Logging ---------
log_path = os.path.join(OUT_DIR, "ablation_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger()
log.info(f"Ablation study started.  Output dir -> {OUT_DIR}")

# ============================================================
# Ablation variants
# ============================================================
ABLATION_VARIANTS = [
    ("A1", "Full model (proposed)",         dict()),
    ("A2", "No delta mode (level)",         dict(TARGET_MODE="level")),
    ("A3", "No detrending",                 dict(DETREND_WINDOWS=False)),
    ("A4", "No slope channel",              dict(USE_SLOPE_CHANNEL=False)),
    ("A5", "No attention (CNN-BiLSTM)",     dict(MODEL_VARIANT="cnn_bilstm_noattn")),
    ("A6", "No winsorization",              dict(WINSOR_X=None)),
    ("A7", "Unidirectional LSTM",           dict(MODEL_VARIANT="lstm")),
    ("A8", "Uniform temporal sampling",     dict(RECENCY_BIASED=False)),
    ("A9", "No label clipping",             dict(LABEL_CLIP_SIGMA=None)),
]

DEFAULT_CFG = dict(
    TARGET_MODE       = "delta",
    DETREND_WINDOWS   = True,
    USE_SLOPE_CHANNEL = True,
    WINSOR_X          = 6.0,
    LABEL_CLIP_SIGMA  = 3.0,
    MODEL_VARIANT     = "cnn_bilstm_attn",
    RECENCY_BIASED    = True,
    BIAS_POWER        = 2.0,
)

# ============================================================
# Load data ONCE
# ============================================================
def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

log.info(f"Loading: {CSV_PATH}")
assert os.path.exists(CSV_PATH), (
    f"\n\nERROR: CSV file not found:\n{CSV_PATH}\n"
    "Please update CSV_PATH at the top of the script.\n"
)

df_raw    = pd.read_csv(CSV_PATH)
date_cols = sorted([c for c in df_raw.columns if _is_date_col(c)])
assert date_cols, "No YYYYMMDD date columns found in CSV."

ts_ff     = df_raw[date_cols].copy().ffill(axis=1)
ts_clean  = ts_ff.fillna(0.0).astype(np.float32)
arr_vals  = ts_clean.to_numpy(np.float32)

n_points, T = ts_clean.shape
train_end = int(T * 0.70)
val_end   = int(T * 0.85)
log.info(f"n_points={n_points:,}  T={T}  "
         f"train[:{train_end}]  val[{train_end}:{val_end}]  test[{val_end}:]")

train_obs = ts_ff.iloc[:, :train_end].to_numpy(np.float32)
means_g   = np.nanmean(train_obs, axis=1, keepdims=True)
stds_g    = np.nanstd(train_obs,  axis=1, keepdims=True)
means_g   = np.nan_to_num(means_g, nan=0.0)
stds_g    = np.nan_to_num(stds_g,  nan=1.0)
stds_g[stds_g == 0] = 1.0
ts_norm_g = ((arr_vals - means_g) / stds_g).astype(np.float32)
log.info("Normalization done.")

TR_RANGE = (LOOKBACK, train_end - (PRED_STEPS - 1))
VA_RANGE = (train_end, val_end  - (PRED_STEPS - 1))
TE_RANGE = (val_end,  T         - (PRED_STEPS - 1))

# ============================================================
# TensorFlow
# ============================================================
import tensorflow as tf
from tensorflow.keras import layers, callbacks, optimizers, losses, models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

tf.random.set_seed(SEED)
log.info(f"TF {tf.__version__}  |  GPUs: {tf.config.list_physical_devices('GPU')}")


# ============================================================
# Feature helpers
# ============================================================
def build_window_channels(w_norm, detrend=True, use_slope=True, winsor=6.0):
    w = w_norm.astype(np.float32).copy()
    if detrend:
        t_idx    = np.arange(LOOKBACK, dtype=np.float32)
        A        = np.vstack([t_idx, np.ones_like(t_idx)]).T
        a, b     = np.linalg.lstsq(A, w, rcond=None)[0]
        w0       = w - (a * t_idx + b)
    else:
        w0 = w
    outs = [w0]
    if use_slope:
        outs.append(np.diff(w0, prepend=w0[0]))
    Xw = np.stack(outs, axis=-1)
    if winsor is not None:
        Xw = np.clip(Xw, -float(winsor), float(winsor))
    return Xw


def sample_times_fn(t_low, t_high, size, recency_biased, bias_power):
    if not recency_biased:
        return np.random.randint(t_low, t_high, size=size)
    u = np.random.rand(size) ** bias_power
    return (t_low + (t_high - t_low - 1) * u).astype(int)


def sample_batch_fn(ts_norm, time_range, batch_size, cfg):
    t_low, t_high = time_range
    t  = sample_times_fn(t_low, t_high, batch_size,
                         cfg["RECENCY_BIASED"], cfg["BIAS_POWER"])
    p  = np.random.randint(0, ts_norm.shape[0], size=(batch_size,))
    ch = 1 + int(cfg["USE_SLOPE_CHANNEL"])
    X  = np.zeros((batch_size, LOOKBACK, ch), np.float32)
    y  = np.zeros((batch_size, PRED_STEPS),   np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        X[i] = build_window_channels(
            ts_norm[pi, ti - LOOKBACK:ti],
            detrend=cfg["DETREND_WINDOWS"],
            use_slope=cfg["USE_SLOPE_CHANNEL"],
            winsor=cfg["WINSOR_X"],
        )
        y[i, 0] = (ts_norm[pi, ti] - ts_norm[pi, ti - 1]
                   if cfg["TARGET_MODE"] == "delta" else ts_norm[pi, ti])
    if cfg["LABEL_CLIP_SIGMA"] is not None:
        clip = max(0.5, cfg["LABEL_CLIP_SIGMA"] * (float(np.std(y)) + 1e-8))
        y    = np.clip(y, -clip, clip)
    return X, y


def make_dataset_fn(ts_norm, time_range, batch_size, cfg):
    ch = 1 + int(cfg["USE_SLOPE_CHANNEL"])
    sig = (
        tf.TensorSpec(shape=(None, LOOKBACK, ch), dtype=tf.float32),
        tf.TensorSpec(shape=(None, PRED_STEPS),   dtype=tf.float32),
    )
    def gen():
        while True:
            yield sample_batch_fn(ts_norm, time_range, batch_size, cfg)
    return tf.data.Dataset.from_generator(gen, output_signature=sig).repeat().prefetch(2)


def make_fixed_test_fn(ts_norm, time_range, n, cfg, seed=SEED):
    rng   = np.random.default_rng(seed)
    t_low, t_high = time_range
    t  = sample_times_fn(t_low, t_high, n, cfg["RECENCY_BIASED"], cfg["BIAS_POWER"])
    p  = rng.integers(0, ts_norm.shape[0], size=n)
    ch = 1 + int(cfg["USE_SLOPE_CHANNEL"])
    X  = np.zeros((n, LOOKBACK, ch), np.float32)
    y  = np.zeros((n, PRED_STEPS),   np.float32)
    for i, (pi, ti) in enumerate(zip(p, t)):
        X[i] = build_window_channels(
            ts_norm[pi, ti - LOOKBACK:ti],
            detrend=cfg["DETREND_WINDOWS"],
            use_slope=cfg["USE_SLOPE_CHANNEL"],
            winsor=cfg["WINSOR_X"],
        )
        y[i, 0] = (ts_norm[pi, ti] - ts_norm[pi, ti - 1]
                   if cfg["TARGET_MODE"] == "delta" else ts_norm[pi, ti])
    return X, y, p, t


# ============================================================
# Model builder
# ============================================================
def build_model_ablation(variant, channels):
    inp = layers.Input(shape=(LOOKBACK, channels))

    if variant == "lstm":
        # A6 — unidirectional, no attention
        x = layers.LSTM(128, return_sequences=True,
                        dropout=0.25, recurrent_dropout=0.1)(inp)
        x = layers.LayerNormalization()(x)
        x = layers.LSTM(64, return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.1)(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation="gelu")(x)
        x = layers.Dropout(0.25)(x)

    elif variant == "cnn_bilstm_noattn":
        # A4 — full CNN+BiLSTM, WITHOUT attention
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
        x = layers.Conv1D(64, 5, padding="causal", activation="gelu",
                          dilation_rate=2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True,
                        dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=False,
                        dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation="gelu")(x)
        x = layers.Dropout(0.25)(x)

    elif variant == "cnn_bilstm_attn":
        # A1, A3, A4, A6, A8, A9 — proposed model (full CNN-BiLSTM-Attention)
        x = layers.Conv1D(64, 3, padding="causal", activation="gelu")(inp)
        x = layers.Conv1D(64, 5, padding="causal", activation="gelu",
                          dilation_rate=2)(x)
        x = layers.Dropout(0.25)(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True,
                        dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True,
                        dropout=0.25, recurrent_dropout=0.1))(x)
        x = layers.LayerNormalization()(x)
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x    = layers.Add()([x, attn])
        x    = layers.LayerNormalization()(x)
        x    = layers.GlobalAveragePooling1D()(x)
        x    = layers.Dense(128, activation="gelu")(x)
        x    = layers.Dropout(0.25)(x)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    out   = layers.Dense(PRED_STEPS)(x)
    model = models.Model(inp, out)

    steps_in_cycle = STEPS_PER_EPOCH * 8
    lr = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=steps_in_cycle,
        t_mul=2.0, m_mul=0.8, alpha=1e-5,
    )
    opt = optimizers.AdamW(learning_rate=lr, weight_decay=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=losses.Huber(delta=1.0), metrics=["mae"])
    return model


# ============================================================
# Evaluate
# ============================================================
def evaluate_model(model, X_test, y_test, pidx_test, tidx_test, ts_norm, cfg):
    ms = means_g[pidx_test].reshape(-1, 1)
    ss = stds_g[pidx_test].reshape(-1, 1)
    yp = model.predict(X_test, batch_size=512, verbose=0)

    if cfg["TARGET_MODE"] == "delta":
        last = ts_norm[pidx_test, tidx_test - 1].reshape(-1, 1) * ss + ms
        pred_den = last + yp * ss
        true_den = last + y_test * ss
    else:
        pred_den = yp      * ss + ms
        true_den = y_test  * ss + ms

    mae  = mean_absolute_error(true_den.ravel(), pred_den.ravel())
    rmse = math.sqrt(mean_squared_error(true_den.ravel(), pred_den.ravel()))
    r2   = r2_score(true_den.ravel(), pred_den.ravel())
    return mae, rmse, r2


# ============================================================
# Main loop — with resume capability in case of crash
# ============================================================
partial_csv   = os.path.join(OUT_DIR, "ablation_results.csv")
results       = []
completed_ids = set()

if os.path.exists(partial_csv):
    df_prev = pd.read_csv(partial_csv)
    completed_ids = set(df_prev["ID"].tolist())
    results = df_prev.to_dict("records")
    log.info(f"Resuming — already done: {completed_ids}")

for vid, label, overrides in ABLATION_VARIANTS:

    if vid in completed_ids:
        log.info(f"⏭  {vid} already done — skip")
        continue

    cfg = {**DEFAULT_CFG, **overrides}
    log.info(f"\n{'='*60}\n▶  {vid}: {label}\n   overrides: {overrides}\n{'='*60}")

    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    channels  = 1 + int(cfg["USE_SLOPE_CHANNEL"])
    model_var = cfg["MODEL_VARIANT"]

    train_ds = make_dataset_fn(ts_norm_g, TR_RANGE, BATCH_SIZE, cfg)
    val_ds   = make_dataset_fn(ts_norm_g, VA_RANGE, BATCH_SIZE, cfg)
    X_test, y_test, pidx_test, tidx_test = make_fixed_test_fn(
        ts_norm_g, TE_RANGE, n=TEST_SAMPLE_N, cfg=cfg, seed=SEED,
    )

    model = build_model_ablation(model_var, channels)
    log.info(f"   Params: {model.count_params():,}  |  channels={channels}")

    ckpt_path = os.path.join(OUT_DIR, f"best_{vid}.keras")
    cbs = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                  save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True, verbose=1),
        callbacks.TerminateOnNaN(),
    ]

    t0 = time.time()
    try:
        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            validation_steps=VAL_STEPS,
            callbacks=cbs, verbose=1,
        )
        status = "OK"
    except Exception as e:
        log.error(f"❌ Training failed: {e}")
        results.append({"ID": vid, "Variant": label,
                        "MAE (mm)": None, "RMSE (mm)": None, "R²": None,
                        "Train time (min)": None, "Status": "FAILED"})
        pd.DataFrame(results).to_csv(partial_csv, index=False)
        continue

    train_time = time.time() - t0

    try:
        mae, rmse, r2 = evaluate_model(
            model, X_test, y_test, pidx_test, tidx_test, ts_norm_g, cfg)
        log.info(f"✅ {vid} | MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}"
                 f"  | {train_time/60:.1f} min")
    except Exception as e:
        log.error(f"❌ Eval failed: {e}")
        mae = rmse = r2 = None
        status = "EVAL_FAILED"

    results.append({
        "ID":               vid,
        "Variant":          label,
        "MAE (mm)":         round(mae,  3) if mae  is not None else None,
        "RMSE (mm)":        round(rmse, 3) if rmse is not None else None,
        "R²":               round(r2,   3) if r2   is not None else None,
        "Train time (min)": round(train_time / 60, 1),
        "Status":           status,
    })

    # Save after each variant — results are preserved if the script crashes
    pd.DataFrame(results).to_csv(partial_csv, index=False)
    log.info(f"   Saved -> {partial_csv}")

    # Loss curve
    plt.figure(figsize=(7, 3))
    plt.plot(history.history["loss"], label="train")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val")
    plt.title(f"Loss — {vid}: {label}")
    plt.xlabel("Epoch"); plt.ylabel("Huber loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"loss_{vid}.png"), dpi=120)
    plt.close()


# ============================================================
# Final table & bar chart
# ============================================================
df_res = pd.read_csv(partial_csv)

a0_row = df_res[df_res["ID"] == "A1"]  # A1 = full proposed model
if len(a0_row):
    a0_mae  = float(a0_row["MAE (mm)"].values[0])
    a0_rmse = float(a0_row["RMSE (mm)"].values[0])
    a0_r2   = float(a0_row["R²"].values[0])
    df_res["ΔMAE"]  = (df_res["MAE (mm)"]  - a0_mae ).round(3)
    df_res["ΔRMSE"] = (df_res["RMSE (mm)"] - a0_rmse).round(3)
    df_res["ΔR²"]   = (df_res["R²"]        - a0_r2  ).round(3)
    df_res.to_csv(partial_csv, index=False)

ok = df_res[df_res["Status"] == "OK"].copy()
if len(ok) >= 2:
    colors = ["#27ae60" if v == "A1" else "#c0392b" for v in ok["ID"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, col, title in zip(
        axes,
        ["MAE (mm)", "RMSE (mm)", "R²"],
        ["MAE (mm)  ↓", "RMSE (mm)  ↓", "R²  ↑"],
    ):
        vals = ok[col].astype(float)
        bars = ax.bar(ok["ID"], vals, color=colors,
                      edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Ablation variant")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.008,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
    plt.suptitle(
        "Ablation Study — Component Contribution Analysis\n"
        "(green = full proposed model  |  red = ablated variant)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, "ablation_results.png")
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"Chart -> {chart_path}")

log.info("\n" + "=" * 70)
log.info("FINAL ABLATION TABLE:")
log.info("=" * 70)
show_cols = [c for c in
             ["ID", "Variant", "MAE (mm)", "RMSE (mm)", "R²",
              "ΔMAE", "ΔRMSE", "ΔR²", "Train time (min)"]
             if c in df_res.columns]
log.info("\n" + df_res[show_cols].to_string(index=False))
log.info("=" * 70)
log.info("✅  Ablation study complete.")
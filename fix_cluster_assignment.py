# -*- coding: utf-8 -*-
"""
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Landslide Deformation using EGMS Time Series"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing 

fix_cluster_assignment.py
==========================
Problem (Andrea's item 3):
  Table 10 (forecasting stratification):  C0=193,009 / C1=165,782 / C2=79,679  / C3=156,508
  Figure 14 (XAI attribution stratification): C0=210,469 / C1=29,869  / C2=89,396  / C3=265,244

Both sum to 594,978 but the per-cluster n's are completely different —
two separate KMeans runs with different random subsets.

Fix: run KMeans ONCE on the full PCA-projected space, assign ALL 594,978
points using the fitted model (predict on the full set), save the
assignment to a .npy file, and report the per-cluster counts.

After running this script:
  1. Use cluster_labels_all.npy as the SINGLE source of truth for
     both Table 10 and Figure 14.
  2. Re-run the forecasting stratification (Table 10) with these labels.
  3. Re-run the XAI stratification (Figure 14) with these labels.

Output:
  cluster_labels_all.npy   — shape (594978,), dtype int32
  cluster_counts.csv       — per-cluster point counts
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ====================== EDIT THESE PATHS IF NEEDED ======================
CSV_PATH = r"E:\خروجی های پروژه\Latronico\EGMS_L2b_044_0234_IW3_VV_2019_2023_1.csv"
OUT_DIR  = r"E:\خروجی های پروژه\outputs_LSTM_BiLSTM\outputs"
# ========================================================================

KMEANS_CLUSTERS = 4
PCA_SAMPLE      = 150_000   # same as original code
RANDOM_STATE    = 0         # same as original code (random_state=0)

def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

print("[INFO] Loading CSV ...")
df = pd.read_csv(CSV_PATH)
date_cols = sorted([c for c in df.columns if _is_date_col(c)])
arr_vals = df[date_cols].fillna(method='ffill', axis=1).fillna(0).to_numpy(np.float32)
n_points = arr_vals.shape[0]
print(f"[INFO] n_points={n_points:,}")

# Standardize each point's trajectory (same as original)
arr_std = (arr_vals - arr_vals.mean(axis=1, keepdims=True)) / \
          (arr_vals.std(axis=1, keepdims=True) + 1e-8)

# PCA on same random subset (same seed)
np.random.seed(RANDOM_STATE)
sel = np.random.choice(n_points, min(PCA_SAMPLE, n_points), replace=False)
print(f"[INFO] Fitting PCA on {len(sel):,} sampled points ...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pc_sample = pca.fit_transform(arr_std[sel])

# KMeans on the sampled PCA projection (same as original)
print(f"[INFO] Fitting KMeans (k={KMEANS_CLUSTERS}) ...")
km = KMeans(n_clusters=KMEANS_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
km.fit(pc_sample)

# ── KEY FIX: assign ALL 594,978 points using km.predict ──────────────
print(f"[INFO] Assigning ALL {n_points:,} points using PCA transform + km.predict ...")
# Transform ALL points through the fitted PCA
pc_all = pca.transform(arr_std)
# Predict cluster for ALL points
labels_all = km.predict(pc_all).astype(np.int32)

# Save
out_labels = os.path.join(OUT_DIR, "cluster_labels_all.npy")
np.save(out_labels, labels_all)
print(f"[SAVED] {out_labels}")

# Report counts
counts = {}
for c in range(KMEANS_CLUSTERS):
    counts[c] = int(np.sum(labels_all == c))

print("\n" + "=" * 55)
print("CLUSTER COUNTS (USE THESE IN BOTH TABLE 10 AND FIGURE 14)")
print("=" * 55)
total = 0
for c, n in counts.items():
    print(f"  Cluster {c}: {n:,} points ({100*n/n_points:.1f}%)")
    total += n
print(f"  TOTAL:     {total:,} points (should be {n_points:,})")
print("=" * 55)

# Save CSV
df_counts = pd.DataFrame([
    {"Cluster": c, "n_points": n, "pct": round(100*n/n_points,1)}
    for c, n in counts.items()
])
out_csv = os.path.join(OUT_DIR, "cluster_counts.csv")
df_counts.to_csv(out_csv, index=False)
print(f"[SAVED] {out_csv}")

# Quick PCA plot colored by new labels (all points, subsample for speed)
plot_idx = np.random.choice(n_points, min(50000, n_points), replace=False)
plt.figure(figsize=(6,5))
plt.scatter(pc_all[plot_idx,0], pc_all[plot_idx,1],
            c=labels_all[plot_idx], s=2, cmap="tab10")
plt.title(f"PCA colored by unified KMeans clusters (k={KMEANS_CLUSTERS}, all {n_points:,} points)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "cluster_labels_pca_check.png")
plt.savefig(out_fig, dpi=150)
plt.close()
print(f"[SAVED] {out_fig}")

print("\nDONE.")
print("\nNEXT STEPS:")
print("  1. Use cluster_labels_all.npy in BOTH the forecasting stratification")
print("     (Table 10) and XAI stratification (Figure 14)")
print("  2. Send cluster_counts.csv to confirm the counts match")

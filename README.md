# Code Repository — Applied Soft Computing (2026)

**Paper:** "Machine Learning and Explainable Deep Learning for Forecasting Landslide Deformation using EGMS Time Series"  
**Authors:** Madadi, Capone, Srinivasareddy, Di Martire, Murano  
**Journal:** Applied Soft Computing  
**Data:** EGMS Level 2b displacement time series — publicly available at https://egms.land.copernicus.eu

---

## Requirements

```
tensorflow >= 2.12
scikit-learn
pandas
numpy
matplotlib
statsmodels
scipy
folium
branca
openpyxl
```

Install with:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib statsmodels scipy folium branca openpyxl
```

---

## Recommended Execution Order

### Step 1 — Train the proposed model
```bash
python LSTM_BiLSTM-M.py
```
Outputs: `outputs/best_lstm.keras`, all visualization figures, `report.html`

> **Config:** Set `CSV_PATH` at the top of the script to point to your EGMS CSV file.  
> **Model variant:** default is `cnn_bilstm_attn` (proposed model). Other options: `lstm`, `bilstm`, `tcn_bilstm_attn`.

---

### Step 2 — Build unified cluster assignment
```bash
python fix_cluster_assignment.py
```
Outputs: `outputs/cluster_labels_all.npy`, `outputs/cluster_counts.csv`

> **Must run before** `recompute_cluster_stratification_v2.py` and `xai_enhanced.py`.  
> This script runs PCA + KMeans once over all 594,978 points and saves a single consistent cluster assignment used by both Table 10 and Figure 14.

---

### Step 3 — Recompute cluster metrics and Figure 14
```bash
python recompute_cluster_stratification_v2.py
```
Outputs: `outputs/table10_cluster_performance_v2.csv`, `outputs/xai_cluster_attribution.png`

> Requires `best_lstm.keras` (Step 1) and `cluster_labels_all.npy` (Step 2).

---

### Step 4 — Run baseline comparisons and DM tests
```bash
# All baselines, 3 seeds (reproduces Table 5 and Table 6):
python baselines_tff.py --multi_seed

# Single seed only:
python baselines_tff.py --seed 42

# Run DM test on already-trained models (no re-training):
python baselines_tff.py --dm_only
```
Outputs: `baseline_outputs/baselines_results.xlsx`, `baseline_outputs/dm_test_*.csv`

> Requires `best_lstm.keras` (Step 1) at `outputs/best_lstm.keras`.

---

### Step 5 — Run ablation study
```bash
python ablation_study.py
```
Outputs: `ablation_outputs/ablation_results.csv`, `ablation_outputs/ablation_results.png`

> Variants A1 (full model) through A9 match Table 7 in the paper exactly.  
> `TEST_SAMPLE_N` is set to 50,000 for CPU compatibility. Set to 200,000 to reproduce exact paper numbers (requires ~16 GB RAM).

---

### Step 6 — XAI attribution analyses
```bash
python xai_enhanced.py
```
Outputs: `xai_cluster_attribution.png` (Figure 14), `xai_activity_attribution.png` (Figure 15), `xai_channel_attribution.png` (Figure 16), `xai_enhanced_summary.xlsx`

> Requires `best_lstm.keras` (Step 1) and `cluster_labels_all.npy` (Step 2).

---

### Step 7 — Persistence baseline (optional verification)
```bash
python fix_persistence_fixed_sample.py
```
Outputs: `outputs/fix_persistence_results.csv`

> Recomputes the Persistence baseline on the exact same fixed 200,000-window test sample used by the proposed model, resolving the small difference between Table 5 (5.234 mm) and Table 8 (5.251 mm), which reflects sampling variance across independently drawn test samples.

---

### Step 8 — Spatial generalization experiment
```bash
python spatial_holdout_test.py
```
Outputs: `outputs/spatial_holdout_results.csv`, `outputs/spatial_holdout_map.png`

> Reproduces Section 5.9 and Table 11. Holds out the easternmost 20% of the corridor by longitude and evaluates the trained model on this geographically distinct block.  
> Requires `best_lstm.keras` (Step 1).

---

## Key Configuration Notes

All scripts share the same core parameters — **these must match across all scripts**:

| Parameter | Value | Description |
|---|---|---|
| `LOOKBACK` | 24 | Input window length (24 × 12-day = ~288 days) |
| `BIAS_POWER` | 2.0 | Recency bias exponent for temporal sampling |
| `SEED` | 42 | Random seed for reproducibility |
| `N_TEST` | 200,000 | Fixed test sample size |
| `TARGET_MODE` | delta | Predict increments, not levels |
| `DETREND_WINDOWS` | True | Local linear detrending within each window |
| `USE_SLOPE_CHANNEL` | True | First-order difference as second input channel |
| `WINSOR_X` | 6.0 | Input winsorization threshold |

---

## Notes on Reproducibility

- All deep learning models use `SEED=42` with `tf.random.set_seed` for reproducibility. Minor numerical differences may arise across hardware configurations.
- Baseline models in `baselines_tff.py` use `SEEDS=[42, 123, 2024]` for multi-seed evaluation (Table 5).
- The cluster assignment in `cluster_labels_all.npy` is the single source of truth for both Table 10 and Figure 14. Do not use the KMeans inside `LSTM_BiLSTM-M.py` for these tables (that KMeans is for visualization only).
- ARIMA results in Table 5 are computed on a subsample of 500 points (walk-forward validation) due to computational constraints; per-sample errors for the DM test are not available for ARIMA.

---

## License

This code is released under the MIT License.

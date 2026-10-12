# APLAND fixed-12-day reproducibility code

This repository contains the analysis code used for the revised manuscript:

> **Machine Learning and Explainable Deep Learning for Forecasting Ground Deformation in a Landslide-Prone EGMS Corridor**  
> Applied Soft Computing (revised manuscript)

The scripts implement the corrected fixed 12-day workflow, including model training, baseline comparison, ablation experiments, uncertainty quantification, PCA--KMeans stratification, coherent XAI, hyperparameter analysis, and the landslide-inventory spatial join.

## 1. Environment

Python **3.10 or 3.11** is recommended. TensorFlow training is computationally intensive; a CUDA-compatible GPU is strongly recommended for the training, baseline, ablation, hyperparameter, MC Dropout, and XAI scripts.

Create and activate a virtual environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The geospatial packages (`geopandas`, `pyogrio`, and `shapely`) are required only for `APLAND_landslide_inventory_join.py`.

## 2. Input data

The raw EGMS L2b CSV is not distributed with this code. Place it at:

```text
data/EGMS_L2b.csv
```

Alternatively, provide its location to each relevant script using `--csv /path/to/file.csv`.

The EGMS file must contain:

- one row per measurement point;
- acquisition columns named in `YYYYMMDD` format;
- point identifiers and spatial coordinates as supplied by EGMS;
- longitude/latitude columns for the landslide-inventory spatial join.

For the inventory analysis, place the complete ESRI Shapefile set at:

```text
data/landslide_inventory/landslides_pol.shp
```

The associated files (for example `.dbf`, `.shx`, `.prj`, and `.cpg`, when available) must remain in the same directory. If the inventory has no embedded CRS, provide it explicitly with `--inventory-crs`, for example `--inventory-crs EPSG:32633`.

Recommended project structure:

```text
project/
├── README.md
├── requirements.txt
├── APLAND_fixed12_*.py
├── APLAND_landslide_inventory_join.py
├── data/
│   ├── EGMS_L2b.csv
│   └── landslide_inventory/
│       ├── landslides_pol.shp
│       └── ... supporting shapefile files ...
└── outputs/                 # created automatically
```

Run all commands from the project root. Relative paths in the scripts are resolved from the current working directory.

## 3. Recommended execution order

### Step 1 — Train and evaluate the final fixed-12-day model

```bash
python APLAND_fixed12_training.py \
  --csv data/EGMS_L2b.csv \
  --output-dir outputs/fixed12_training
```

Main outputs include `best_fixed12.keras`, `test_predictions_fixed12.csv`, `metrics_fixed12.json`, cadence audits, and the training history. The default seed is 42 and the fixed test sample contains 200,000 point--time windows.

An optional cadence-only audit can be run without model training:

```bash
python APLAND_fixed12_training.py --csv data/EGMS_L2b.csv --cadence-only
```

### Step 2 — Recompute the persistence baseline on the identical test sample

```bash
python APLAND_fixed12_persistence_audit.py \
  --predictions outputs/fixed12_training/test_predictions_fixed12.csv \
  --output-dir outputs/fixed12_persistence_audit
```

This script uses the saved test predictions and does not redraw samples.

### Step 3 — Create the unified PCA--KMeans labels

```bash
python APLAND_fixed12_unified_clusters.py \
  --csv data/EGMS_L2b.csv \
  --output-dir outputs/fixed12_clustering
```

The explicit output directory above is important because the downstream scripts expect `outputs/fixed12_clustering/cluster_labels_all.npy`. PCA is fitted on a fixed sample of at most 150,000 standardized trajectories; all trajectories are projected, and the KMeans model is then fitted once to the complete set of PCA projections.

### Step 4 — Compute cluster-stratified test metrics

```bash
python APLAND_fixed12_cluster_stratification.py \
  --predictions outputs/fixed12_training/test_predictions_fixed12.csv \
  --labels outputs/fixed12_clustering/cluster_labels_all.npy \
  --output-dir outputs/fixed12_cluster_stratification
```

### Step 5 — Run MC Dropout uncertainty quantification

```bash
python APLAND_fixed12_mc_dropout.py \
  --csv data/EGMS_L2b.csv \
  --base-script APLAND_fixed12_training.py \
  --model outputs/fixed12_training/best_fixed12.keras \
  --predictions outputs/fixed12_training/test_predictions_fixed12.csv \
  --output-dir outputs/fixed12_mc_dropout
```

The default analysis uses 100 stochastic forward passes. Interval calibration is performed on a fixed validation sample, and coverage is evaluated on the unchanged test sample.

### Step 6 — Run coherent XAI and fidelity analyses

```bash
python APLAND_fixed12_coherent_xai.py \
  --csv data/EGMS_L2b.csv \
  --model outputs/fixed12_training/best_fixed12.keras \
  --predictions outputs/fixed12_training/test_predictions_fixed12.csv \
  --labels outputs/fixed12_clustering/cluster_labels_all.npy \
  --output-dir outputs/fixed12_xai
```

This produces Integrated Gradients and SmoothIG summaries, completeness checks, and coherent deletion/insertion curves with attribution-ranked, random, and least-important-first controls. The slope channel is recomputed after each displacement-trajectory perturbation.

### Step 7 — Run the multi-seed baseline comparison and DM analysis

```bash
python APLAND_fixed12_baselines_multiseed.py \
  --csv data/EGMS_L2b.csv \
  --output-dir outputs/fixed12_baselines \
  --multi-seed
```

The three seeds are 42, 123, and 2024. The script evaluates the directly comparable models under the shared fixed-12-day protocol and generates the epoch-aggregated Diebold--Mariano results with the Harvey--Leybourne--Newbold correction. ARIMA is implemented as a separate walk-forward experiment within this script.

To recompute only the DM tables from previously saved error files:

```bash
python APLAND_fixed12_baselines_multiseed.py \
  --csv data/EGMS_L2b.csv \
  --output-dir outputs/fixed12_baselines \
  --statistics-only
```

### Step 8 — Run the A1--A9 multi-seed ablation study

```bash
python APLAND_fixed12_ablation_multiseed.py \
  --csv data/EGMS_L2b.csv \
  --output-dir outputs/fixed12_ablation \
  --variants A1,A2,A3,A4,A5,A6,A7,A8,A9 \
  --seeds 42,123,2024
```

The output includes raw per-run metrics and the mean ± standard-deviation summary.

### Step 9 — Run the limited validation-only hyperparameter search

```bash
python APLAND_fixed12_hyperparameter_search.py \
  --csv data/EGMS_L2b.csv \
  --base-script APLAND_fixed12_training.py \
  --output-dir outputs/fixed12_hyperparameter_search
```

The seven prespecified configurations H1--H7 compare lookback length, Huber-loss delta, and recent-sampling bias power. Selection uses the common validation sample; test labels are not used for hyperparameter selection. Add `--force` only when completed configurations must intentionally be rerun.

### Step 10 — Join EGMS points to the landslide inventory

```bash
python APLAND_landslide_inventory_join.py \
  --csv data/EGMS_L2b.csv \
  --shapefile data/landslide_inventory/landslides_pol.shp \
  --output-dir outputs/landslide_inventory_join \
  --predictions outputs/fixed12_training/test_predictions_fixed12.csv
```

The `--predictions` argument is optional and adds within-/outside-inventory performance summaries. The script expects the EGMS point coordinates in WGS84 longitude/latitude and reprojects the points to the inventory CRS before the spatial join.

## 4. Script summary

| Script | Purpose | Main prerequisites |
|---|---|---|
| `APLAND_fixed12_training.py` | Final model training and test evaluation | EGMS CSV |
| `APLAND_fixed12_persistence_audit.py` | Persistence baseline on the exact saved test sample | Training predictions |
| `APLAND_fixed12_unified_clusters.py` | Reproducible PCA--KMeans labels | EGMS CSV |
| `APLAND_fixed12_cluster_stratification.py` | Cluster-wise test performance | Training predictions and cluster labels |
| `APLAND_fixed12_mc_dropout.py` | Calibrated MC Dropout intervals | Training script, trained model, predictions, EGMS CSV |
| `APLAND_fixed12_coherent_xai.py` | IG/SmoothIG and coherent fidelity tests | Trained model, predictions, cluster labels, EGMS CSV |
| `APLAND_fixed12_baselines_multiseed.py` | Baselines, multi-seed results, and epoch-aggregated DM tests | EGMS CSV |
| `APLAND_fixed12_ablation_multiseed.py` | A1--A9 multi-seed ablation study | EGMS CSV |
| `APLAND_fixed12_hyperparameter_search.py` | Seven-configuration validation search | Training script and EGMS CSV |
| `APLAND_landslide_inventory_join.py` | EGMS--landslide inventory spatial overlay | EGMS CSV and complete inventory shapefile |

## 5. Reproducibility notes

- Keep the same raw EGMS CSV and the same optional `--anchor` across all scripts.
- Do not reorder the rows of the EGMS CSV after producing `cluster_labels_all.npy`; cluster labels are aligned to the original row indices.
- Downstream analyses reuse `test_predictions_fixed12.csv` and must not redraw the fixed test sample.
- Preserve the default test-sample seed when comparing training seeds.
- The output directories may require substantial disk space because trained models, per-window predictions, and error arrays are saved.
- The scripts create output directories automatically and will fail with explicit messages when required inputs are missing.

## 6. Quick interface check

The command-line options for any script can be displayed with:

```bash
python SCRIPT_NAME.py --help
```

This checks the command-line interface only; it does not run the experiment.


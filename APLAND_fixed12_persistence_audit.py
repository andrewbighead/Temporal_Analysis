#!/usr/bin/env python3
"""
Paper: "Machine Learning and Explainable Deep Learning for Forecasting Ground
        Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

APLAND_fixed12_persistence_audit.py
====================================
Recomputes Persistence directly from the proposed model's saved fixed-12-day
test predictions. It never redraws samples, reloads a model, or reconstructs
the raw time axis, so both models are evaluated on exactly the same rows.

Input: test_predictions_fixed12.csv from APLAND_fixed12_training.py
Outputs: persistence_fixed12_metrics.csv, persistence_fixed12_metrics.tex,
         persistence_audit.json

Example
-------
python APLAND_fixed12_persistence_audit.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = {
    "point_index", "target_epoch_index", "target_date", "last_observed_date",
    "horizon_days", "true_increment_mm", "predicted_increment_mm",
    "last_observed_level_mm", "true_level_mm", "predicted_level_mm",
}
def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions", type=Path,
        default=Path("outputs/fixed12_training/test_predictions_fixed12.csv"),
        help="Fixed test predictions produced by APLAND_fixed12_training.py.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/fixed12_persistence_audit"),
    )
    p.add_argument("--expected-samples", type=int, default=200_000)
    return p.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_pred - y_true
    denom = np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64)
    r2 = 1.0 - np.sum(error ** 2, dtype=np.float64) / denom if denom > 0 else math.nan
    return {
        "MAE_mm": float(np.mean(np.abs(error))),
        "RMSE_mm": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(r2),
    }


def main() -> None:
    args = arguments()
    pred_path = args.predictions.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    if args.expected_samples < 1:
        raise ValueError("--expected-samples must be positive.")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path, low_memory=False)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    if len(df) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples:,} rows, found {len(df):,}")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Prediction file contains missing values in required columns")
    horizons = sorted(df["horizon_days"].astype(int).unique().tolist())
    if horizons != [12]:
        raise ValueError(f"Not a fixed 12-day test set; horizons found: {horizons}")
    target_dates = pd.to_datetime(df["target_date"], errors="raise")
    previous_dates = pd.to_datetime(df["last_observed_date"], errors="raise")
    if not np.all((target_dates - previous_dates).dt.days.to_numpy() == 12):
        raise ValueError("At least one target/previous-date pair is not 12 days apart")
    if (df["point_index"].astype(int) < 0).any():
        raise ValueError("Prediction file contains a negative point index.")

    true_delta = df["true_increment_mm"].to_numpy(np.float64)
    proposed_delta = df["predicted_increment_mm"].to_numpy(np.float64)
    true_level = df["true_level_mm"].to_numpy(np.float64)
    proposed_level = df["predicted_level_mm"].to_numpy(np.float64)
    last_level = df["last_observed_level_mm"].to_numpy(np.float64)
    persistence_delta = np.zeros(len(df), dtype=np.float64)
    persistence_level = last_level

    # These identities must hold for delta-mode reconstruction. Checking them
    # prevents silently auditing a prediction file from another experiment.
    if not np.allclose(true_level, last_level + true_delta, atol=1e-4, rtol=1e-6):
        raise ValueError("True reconstructed levels are inconsistent with increments.")
    if not np.allclose(
        proposed_level, last_level + proposed_delta, atol=1e-4, rtol=1e-6
    ):
        raise ValueError("Predicted reconstructed levels are inconsistent with increments.")

    rows = []
    for name, pred_d, pred_l in [
        ("CNN-BiLSTM-Attention (proposed)", proposed_delta, proposed_level),
        ("Persistence / zero-increment", persistence_delta, persistence_level),
    ]:
        md, ml = metrics(true_delta, pred_d), metrics(true_level, pred_l)
        rows.append({"Model": name,
                     "MAE_delta_mm": md["MAE_mm"], "RMSE_delta_mm": md["RMSE_mm"],
                     "R2_delta": md["R2"], "MAE_level_mm": ml["MAE_mm"],
                     "RMSE_level_mm": ml["RMSE_mm"], "R2_level": ml["R2"]})
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "persistence_fixed12_metrics.csv", index=False)

    # In delta-mode, adding the same last level to truth and prediction leaves
    # every error unchanged; MAE and RMSE must therefore match in both spaces.
    for _, row in results.iterrows():
        if not np.isclose(row["MAE_delta_mm"], row["MAE_level_mm"], atol=1e-10):
            raise AssertionError("Delta- and level-space MAE should be identical.")
        if not np.isclose(row["RMSE_delta_mm"], row["RMSE_level_mm"], atol=1e-10):
            raise AssertionError("Delta- and level-space RMSE should be identical.")

    # LaTeX table for paper
    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Persistence audit on the exact fixed 12-day test sample.}",
        r"\label{tab:persistence_audit}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & MAE$_\Delta$ (mm) & RMSE$_\Delta$ (mm) & $R^2_\Delta$ & $R^2_{\text{level}}$ \\",
        r"\midrule",
    ]
    for _, row in results.iterrows():
        latex_lines.append(
            f"{row['Model']} & {row['MAE_delta_mm']:.3f} & {row['RMSE_delta_mm']:.3f} "
            f"& {row['R2_delta']:.3f} & {row['R2_level']:.3f} \\\\"
        )
    latex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out_dir / "persistence_fixed12_metrics.tex").write_text(
        "\n".join(latex_lines), encoding="utf-8"
    )

    persistence_row = results.loc[
        results["Model"] == "Persistence / zero-increment"
    ].iloc[0]
    audit = {
        "source_predictions": str(args.predictions), "sample_count": len(df),
        "unique_point_count": int(df["point_index"].nunique()),
        "unique_target_epoch_count": int(df["target_epoch_index"].nunique()),
        "horizon_days": horizons, "sampling_action": "none; reused saved model test rows",
        "persistence_definition": "predicted increment = 0; predicted level = last observed level",
        "mean_true_increment_mm": float(np.mean(true_delta)),
        "persistence_mae_mm": float(persistence_row["MAE_level_mm"]),
        "persistence_rmse_mm": float(persistence_row["RMSE_level_mm"]),
        "persistence_r2_level": float(persistence_row["R2_level"]),
        "persistence_r2_delta": float(persistence_row["R2_delta"]),
    }
    (out_dir / "persistence_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nEXACT SHARED FIXED-12-DAY TEST RESULTS")
    print(results.to_string(index=False))
    print(f"\n[DONE] Outputs written to {out_dir}")

if __name__ == "__main__":
    main()

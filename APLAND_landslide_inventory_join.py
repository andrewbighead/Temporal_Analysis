#!/usr/bin/env python3
"""
APLAND_landslide_inventory_join.py
====================================
Paper: "Machine Learning and Explainable Deep Learning for Forecasting
        Ground Deformation in a Landslide-Prone EGMS Corridor"
Authors: Madadi, Capone, Srinivasareddy, Di Martire, Murano
Journal: Applied Soft Computing (manuscript under revision)

Spatial join between EGMS measurement points and the Basilicata
landslide inventory (Lazzari & Gioia, refs [36] and [39]).

Usage:
  python APLAND_landslide_inventory_join.py
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── helpers ───────────────────────────────────────────────────────────────────
def _is_date_col(c):
    return bool(re.fullmatch(r"\d{8}", str(c)))

def arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/EGMS_L2b.csv"))
    ap.add_argument(
        "--shapefile", type=Path,
        default=Path("data/landslide_inventory/landslides_pol.shp"),
        help="Path to the main .shp file of the landslide inventory.",
    )
    ap.add_argument(
        "--inventory-crs", default=None,
        help="CRS to assign only if the inventory has no CRS metadata (e.g. EPSG:32633).",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/landslide_inventory_join"),
    )
    ap.add_argument("--predictions", type=Path, default=None,
                    help="Optional: test_predictions_fixed12.csv to report "
                         "per-group performance")
    return ap.parse_args()


def main() -> None:
    args = arguments()

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "This spatial audit requires GeoPandas. Install the geospatial "
            "dependencies with: pip install geopandas pyogrio shapely"
        ) from exc

    csv_path = args.csv.expanduser().resolve()
    inventory_path = args.shapefile.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"EGMS CSV not found: {csv_path}")
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Inventory shapefile not found: {inventory_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load EGMS points ──────────────────────────────────────────────────────
    print("[1/4] Loading EGMS CSV...")
    df = pd.read_csv(csv_path, low_memory=False)

    lat_col = next((c for c in df.columns
                    if str(c).lower() in ["lat", "latitude"]), None)
    lon_col = next((c for c in df.columns
                    if str(c).lower() in ["lon", "longitude"]), None)

    if lat_col is None or lon_col is None:
        non_date = [c for c in df.columns if not _is_date_col(c)]
        print(f"  Available non-date columns: {non_date[:10]}")
        raise ValueError(
            f"Could not find lat/lon columns. "
            f"Available: {df.columns.tolist()[:15]}"
        )

    lats = pd.to_numeric(df[lat_col], errors="raise").to_numpy(np.float64)
    lons = pd.to_numeric(df[lon_col], errors="raise").to_numpy(np.float64)
    if not np.isfinite(lats).all() or not np.isfinite(lons).all():
        raise ValueError("Latitude/longitude columns contain non-finite values.")
    if np.any((lats < -90) | (lats > 90)) or np.any((lons < -180) | (lons > 180)):
        raise ValueError("Coordinates are outside valid WGS84 longitude/latitude ranges.")
    n_pts = len(df)
    print(f"  n_points = {n_pts:,}")
    print(f"  lat range: [{lats.min():.4f}, {lats.max():.4f}]")
    print(f"  lon range: [{lons.min():.4f}, {lons.max():.4f}]")

    # ── load shapefile ────────────────────────────────────────────────────────
    print("\n[2/4] Loading landslide inventory shapefile...")
    print(f"  Loading: {inventory_path}")
    inv = gpd.read_file(inventory_path)
    if inv.empty:
        raise ValueError("The landslide inventory contains no features.")
    inv = inv.loc[inv.geometry.notna() & ~inv.geometry.is_empty].copy()
    if inv.empty:
        raise ValueError("The landslide inventory contains no usable geometries.")
    if not inv.geometry.is_valid.all():
        inv.geometry = inv.geometry.make_valid()
    inventory_crs_original = str(inv.crs) if inv.crs is not None else None
    print(f"  CRS: {inv.crs}")
    print(f"  n_polygons: {len(inv)}")
    print(f"  Columns: {inv.columns.tolist()}")
    print(f"  Bounds: {inv.total_bounds}")

    # ── create GeoDataFrame for EGMS points ───────────────────────────────────
    print("\n[3/4] Performing spatial join...")
    gdf_pts  = gpd.GeoDataFrame(
        {"point_index": np.arange(n_pts),
         "lat": lats, "lon": lons},
        geometry=gpd.points_from_xy(lons, lats),
        crs="EPSG:4326"
    )

    # reproject inventory to WGS84 if needed
    if inv.crs is None:
        if args.inventory_crs is None:
            raise ValueError(
                "The inventory has no CRS metadata. Supply its documented CRS "
                "with --inventory-crs; the script will not assume one."
            )
        inv = inv.set_crs(args.inventory_crs)
    elif inv.crs.to_epsg() != 4326:
        inv = inv.to_crs("EPSG:4326")

    # spatial join
    joined = gpd.sjoin(
        gdf_pts[["point_index", "geometry"]], inv[["geometry"]],
        how="inner", predicate="within"
    )
    # A point can intersect overlapping inventory polygons. Count each EGMS
    # point exactly once, independently of polygon overlap.
    matched_point_indices = np.sort(joined["point_index"].unique().astype(np.int64))
    in_landslide = np.zeros(n_pts, dtype=bool)
    in_landslide[matched_point_indices] = True
    n_in  = int(in_landslide.sum())
    n_out = n_pts - n_in
    pct_in  = n_in  / n_pts * 100
    pct_out = n_out / n_pts * 100

    print(f"\n  Points INSIDE landslide polygons:  {n_in:,} ({pct_in:.1f}%)")
    print(f"  Points OUTSIDE landslide polygons: {n_out:,} ({pct_out:.1f}%)")
    print(f"  Total:                             {n_pts:,} (100%)")

    # ── optional: per-group performance ──────────────────────────────────────
    if args.predictions is not None:
        print("\n[3b/4] Computing per-group forecasting performance...")
        predictions_path = args.predictions.expanduser().resolve()
        if not predictions_path.is_file():
            raise FileNotFoundError(f"Prediction file not found: {predictions_path}")
        pred_df = pd.read_csv(predictions_path)
        required_prediction_columns = {
            "point_index", "true_level_mm", "predicted_level_mm",
            "true_increment_mm", "predicted_increment_mm",
        }
        missing = required_prediction_columns.difference(pred_df.columns)
        if missing:
            raise ValueError(f"Prediction file lacks columns: {sorted(missing)}")
        point_indices = pred_df["point_index"].to_numpy(np.int64)
        if np.any(point_indices < 0) or np.any(point_indices >= n_pts):
            raise ValueError("Prediction file contains an invalid point index.")

        # map point_index → in/out
        pred_df["in_landslide"] = in_landslide[point_indices]

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        results = {}
        for grp, label in [(True, "Inside landslide"), (False, "Outside landslide")]:
            mask = pred_df["in_landslide"] == grp
            sub  = pred_df[mask]
            if len(sub) == 0:
                continue
            mae   = mean_absolute_error(sub["true_level_mm"],
                                        sub["predicted_level_mm"])
            rmse  = float(np.sqrt(mean_squared_error(
                sub["true_level_mm"], sub["predicted_level_mm"])))
            r2l   = r2_score(sub["true_level_mm"], sub["predicted_level_mm"])
            r2d   = r2_score(sub["true_increment_mm"],
                             sub["predicted_increment_mm"])
            results[label] = {
                "n_windows": int(len(sub)),
                "n_unique_points": int(sub["point_index"].nunique()),
                "MAE_level_mm": round(mae, 3),
                "RMSE_level_mm": round(rmse, 3),
                "R2_level": round(r2l, 3),
                "R2_delta": round(r2d, 3),
            }
            print(f"\n  {label}:")
            print(f"    n_windows={len(sub):,}  n_points={sub['point_index'].nunique():,}")
            print(f"    MAE={mae:.3f}mm  RMSE={rmse:.3f}mm  "
                  f"R²_level={r2l:.3f}  R²_delta={r2d:.3f}")
    else:
        results = {}

    # ── save audit ────────────────────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    audit = {
        "total_egms_points": n_pts,
        "points_inside_landslide_polygons": n_in,
        "points_outside_landslide_polygons": n_out,
        "pct_inside": round(pct_in, 2),
        "pct_outside": round(pct_out, 2),
        "spatial_predicate": "within",
        "overlapping_polygon_matches_deduplicated_by_point_index": True,
        "inventory_source": str(args.shapefile),
        "inventory_crs_original": inventory_crs_original,
        "analysis_crs": "EPSG:4326",
        "inventory_n_polygons": len(inv),
        "performance_by_group": results,
    }
    audit_path = out_dir / "landslide_inventory_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"[SAVED] {audit_path}")

    pd.DataFrame({
        "point_index": np.arange(n_pts, dtype=np.int64),
        "longitude": lons,
        "latitude": lats,
        "within_mapped_landslide_polygon": in_landslide,
    }).to_csv(out_dir / "egms_landslide_membership.csv", index=False)

    # ── Figure: map ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    inv.boundary.plot(ax=ax, color="red", linewidth=0.8,
                      label=f"Landslide polygons (n={len(inv)})")
    ax.scatter(lons[~in_landslide],
               lats[~in_landslide],
               s=0.5, c="gray", alpha=0.3, label=f"Outside ({n_out:,})")
    ax.scatter(lons[in_landslide],
               lats[in_landslide],
               s=1.0, c="orange", alpha=0.6,
               label=f"Inside landslide ({n_in:,}, {pct_in:.1f}%)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"EGMS points vs Basilicata landslide inventory\n"
                 f"{n_in:,}/{n_pts:,} points ({pct_in:.1f}%) within mapped landslide bodies")
    ax.legend(markerscale=5, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = out_dir / "landslide_inventory_map.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {fig_path}")

    # ── print summary for paper ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY FOR PAPER")
    print("=" * 65)
    print(f"\nOf the {n_pts:,} EGMS measurement points in the study corridor,")
    print(f"{n_in:,} ({pct_in:.1f}%) fall within mapped landslide bodies")
    print(f"according to the Basilicata landslide inventory")
    print(f"(Lazzari & Gioia, refs [36] and [39]),")
    print(f"while {n_out:,} ({pct_out:.1f}%) are located outside mapped landslide areas.")
    print("\nDONE.")


if __name__ == "__main__":
    main()

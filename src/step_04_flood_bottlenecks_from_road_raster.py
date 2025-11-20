"""
Step 04 — Flood-aware bottlenecks (1-km raster screening)

Goal
-----
Flag 1-km cells where **main OSM roads** intersect **RP100 flood depth >= threshold**,
and identify which of those risky cells are **near priority** areas.

Reads (AOI-prefixed / previous steps)
-------------------------------------
- PATHS.TRAVEL                      (target grid; minutes)
- FLOOD1K_TIF        = {AOI}_flood_rp100_maxdepth_1km.tif        (from Step 00)
- PRIORITY_TOP10_TIF  = {AOI}_priority_top10_mask.tif            (from Step 03)
- PATHS.ROADS         (OSM lines; rasterized here)

Writes
------
- ROADS1K_TIF          = {AOI}_roads_main_1km.tif
- ROADS_RISK_TIF       = {AOI}_roads_flood_risk_cells_1km.tif
- ROADS_RISK_NEAR_TIF  = {AOI}_roads_flood_risk_near_priority_1km.tif
- tables/{AOI}_roads_flood_risk_summary.csv

Notes
-----
- This is a **screening** method at 1-km resolution (cells ≈ km). For engineering-grade
  lengths at risk, run a vector intersect (roads × flood depth) and measure true line lengths.
"""

from __future__ import annotations
from pathlib import Path

import rioxarray as rxr
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import maximum_filter
from rasterio.enums import Resampling
from config import (
    AOI, PATHS, PARAMS, ROADS1K_TIF, FLOOD1K_TIF,
    PRIORITY_TOP10_TIF_V1, ROADS_RISK_TIF, ROADS_RISK_NEAR_TIF,
    out_r, out_t, get_logger
)
from utils_geo import (
    open_template, rasterize_vector, 
    write_gtiff, write_gtiff_masked,
    estimate_cell_area_km2
)

log = get_logger(__name__)


def _xr_from_numpy(arr: np.ndarray, like: xr.DataArray) -> xr.DataArray:
    """Rebuild a DataArray from a numpy array matching `like`'s georeferencing."""
    da = xr.DataArray(arr, coords={"y": like.y, "x": like.x}, dims=("y", "x"))
    da.rio.write_crs(like.rio.crs, inplace=True)
    da.rio.write_transform(like.rio.transform(), inplace=True)
    return da


def _assert_same_shape(*das: xr.DataArray) -> None:
    """Raise AssertionError if rasters do not share identical shape."""
    shapes = {da.shape for da in das}
    assert len(shapes) == 1, f"Rasters must share identical shape; got {shapes}"


def main() -> None:
    # -------------------------------------------------------------------------
    # Load target grid and inputs, and ensure they match the target grid
    # -------------------------------------------------------------------------
    T = open_template(PATHS.TRAVEL)
    flood1k = open_template(FLOOD1K_TIF)
    prio10  = open_template(PRIORITY_TOP10_TIF_V1)

    # Optional: precomputed flood exceedance fraction (0..1) from Step 00
    flood_frac = None
    frac_path = out_r("flood_rp100_exceed_frac_1km")
    if Path(frac_path).exists():
        flood_frac = rxr.open_rasterio(frac_path, masked=True).squeeze()

    # If anything doesn't match T, reproject it on the fly
    fixed = []
    if flood1k.shape != T.shape or flood1k.rio.transform() != T.rio.transform() or flood1k.rio.crs != T.rio.crs:
        flood1k = flood1k.rio.reproject_match(T, resampling=Resampling.max)
        fixed.append("flood1k")
    
    if flood_frac is not None:
        if (flood_frac.shape != T.shape) or (flood_frac.rio.transform() != T.rio.transform()) or (flood_frac.rio.crs != T.rio.crs):
            flood_frac = flood_frac.rio.reproject_match(T, resampling=Resampling.average)
            fixed.append("flood_frac")

    if prio10.shape != T.shape or prio10.rio.transform() != T.rio.transform() or prio10.rio.crs != T.rio.crs:
        # nearest is appropriate for binary masks
        prio10 = prio10.rio.reproject_match(T, resampling=Resampling.nearest)
        fixed.append("prio10")

    if fixed:
        log.warning(f"Reprojected to match target grid: {', '.join(fixed)}")

    # Now it’s safe to assert shapes match
    _assert_same_shape(T, flood1k, prio10)

    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"Flood screening inputs loaded | CRS={T.rio.crs} | "
        f"size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}"
    )
    rc = getattr(PARAMS, "ROAD_CLASSES_KEEP", None)
    rc_pretty = rc if (rc is None) else tuple(rc)

    # define this BEFORE logging
    roads_all_touched = bool(getattr(PARAMS, "ROADS_ALL_TOUCHED", False))  # default False for stricter counts
    
    log.info(
        f"Params | FLOOD_DEPTH_RISK={PARAMS.FLOOD_DEPTH_RISK} m | "
        f"ROAD_CLASSES_KEEP={rc_pretty} | ROADS_ALL_TOUCHED={roads_all_touched}"
    )

    # -------------------------------------------------------------------------
    # Rasterize main OSM road classes to 1-km (presence=1)
    # all_touched=True helps capture thin lines that just graze a cell.
    # Policy:
    #   • ROAD_CLASSES_KEEP is None or empty  -> NO FILTER (use all fclass values)
    #   • Otherwise                           -> filter where fclass in ROAD_CLASSES_KEEP
    # -------------------------------------------------------------------------
    keep_raw = getattr(PARAMS, "ROAD_CLASSES_KEEP", None)

    # Decide if we filter or not
    no_filter = (keep_raw is None) or (isinstance(keep_raw, (list, tuple)) and len(keep_raw) == 0)

    if no_filter:
        where = None
        log.info("Rasterizing roads: no class filter (use all OSM fclass values).")
    else:
        # Build a safe OR clause like: fclass == 'primary' or fclass == 'secondary' ...
        classes = tuple(str(c).replace("'", "''") for c in keep_raw)  # escape single quotes
        where = " or ".join([f"fclass == '{c}'" for c in classes])
        log.info(f"Rasterizing roads where: {where}")

    roads1k = rasterize_vector(PATHS.ROADS, T, where=where, burn_value=1, all_touched=roads_all_touched)

    write_gtiff(roads1k, ROADS1K_TIF, like=T)

    road_cells = int(np.nansum(roads1k.values == 1))
    log.info(f"Wrote {ROADS1K_TIF.name} | road_cells={road_cells:,}")

    if road_cells == 0:
        log.warning(
            "No road cells rasterized. If you expected roads:\n"
            "  • Ensure PATHS.ROADS points to an OSM lines file with an 'fclass' field;\n"
            "  • Leave ROAD_CLASSES_KEEP as None to disable filtering; or\n"
            "  • Provide exact OSM classes (e.g., ('primary','secondary','tertiary'))."
        )
    
    # Continue anyway; risk layers will be all zeros.
    # -------------------------------------------------------------------------
    # Risk cells: road presence AND flood depth >= threshold
    # -------------------------------------------------------------------------
    # Risk cells:
    # Prefer fraction-based screening if available; otherwise fall back to depth≥threshold.
    fmin = float(getattr(PARAMS, "FLOOD_EXCEED_FRACTION_MIN", 0.25))  # policy knob; default 25%
    if flood_frac is not None:
        method_note = f"fraction≥{fmin:.2f}"
        risk_cells = ((roads1k == 1) & (flood_frac >= fmin)).astype("int16")
    else:
        method_note = f"depth≥{float(PARAMS.FLOOD_DEPTH_RISK)}m (fallback)"
        risk_cells = ((roads1k == 1) & (flood1k >= PARAMS.FLOOD_DEPTH_RISK)).astype("int16")

    write_gtiff_masked(risk_cells, ROADS_RISK_TIF, like=T, nodata=np.nan)
    risk_total = int(np.nansum(risk_cells.values))
    log.info(f"Wrote {ROADS_RISK_TIF.name} | risk_cells={risk_total:,}")

    # -------------------------------------------------------------------------
    # “Near priority”: 1-cell (~1 km) dilation on priority mask, intersect with risk
    # -------------------------------------------------------------------------
    prio_buf = maximum_filter(np.nan_to_num(prio10.values, nan=0), size=3)
    prio_prox = ((prio_buf >= 1) & (risk_cells.values == 1)).astype("int16")
    prio_prox_da = _xr_from_numpy(prio_prox, like=T)
    write_gtiff_masked(prio_prox_da, ROADS_RISK_NEAR_TIF, like=T, nodata=np.nan)
    risk_near = int(np.nansum(prio_prox))
    log.info(f"Wrote {ROADS_RISK_NEAR_TIF.name} | risk_near_priority_cells={risk_near:,}")

    # -------------------------------------------------------------------------
    # Summary table (add context: road cells, percentages, params, timestamp)
    # -------------------------------------------------------------------------
    dt_utc = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Defensive counts (cells)
    tot_road      = int(np.nan_to_num(roads1k.values,      nan=0).sum())
    tot_risk      = int(np.nan_to_num(risk_cells.values,   nan=0).sum())
    tot_near_prio = int(np.nan_to_num(prio_prox,           nan=0).sum())

    # Percentages
    pct_at_risk        = (100.0 * tot_risk / tot_road) if tot_road > 0 else np.nan
    pct_near_of_risk   = (100.0 * tot_near_prio / tot_risk) if tot_risk > 0 else np.nan

    # Pretty print the road filter actually applied
    keep_raw = getattr(PARAMS, "ROAD_CLASSES_KEEP", None)
    road_filter_applied = "ALL" if (keep_raw is None or len(keep_raw) == 0) else ",".join(map(str, keep_raw))

    # Estimate km² per cell (mean), robust to array/scalar return
    try:
        cell_km2_est = estimate_cell_area_km2(T)
        cell_km2 = float(np.nanmean(cell_km2_est)) if hasattr(cell_km2_est, "shape") else float(cell_km2_est)
    except Exception:
        cell_km2 = np.nan

    summary = pd.DataFrame([{
        "aoi": AOI,
        "date_utc": dt_utc,
        "cell_km2": cell_km2,

        "total_road_cells": int(tot_road),
        "total_risk_cells": int(tot_risk),
        "risk_pct_of_roads": round(pct_at_risk, 2) if np.isfinite(pct_at_risk) else np.nan,

        "risk_near_priority_cells": int(tot_near_prio),
        "near_prio_pct_of_risk": round(pct_near_of_risk, 2) if np.isfinite(pct_near_of_risk) else np.nan,

        "flood_depth_threshold_m": float(PARAMS.FLOOD_DEPTH_RISK),
        "road_filter_applied": road_filter_applied,
        "flood_exceed_fraction_min": (fmin if flood_frac is not None else np.nan),

        "notes": (
            "Risk=roads ∩ flood; near-priority=within 1 cell of Top10% priority; "
            f"method={method_note}; roads_all_touched={roads_all_touched}"
        )
    }])

    summary_path = out_t("roads_flood_risk_summary")
    summary.to_csv(summary_path, index=False)
    log.info(f"Saved summary → {summary_path}")

    log.info("Step 04 complete.")


if __name__ == "__main__":
    main()

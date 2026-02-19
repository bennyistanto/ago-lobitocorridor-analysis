"""
Step 16 — Intervention Impact Simulator

Purpose
-------
Answer "What if we upgrade road X?" by recomputing the travel-time surface
after a hypothetical intervention and quantifying the population-level impact.

How it works
------------
1. Load the friction surface built in Step 12 (minutes per km).
2. For each proposed intervention (road segment), reduce friction on cells that
   the segment traverses to simulate an upgrade (e.g., track → paved secondary).
3. Recompute minimum-cost travel from *all project sites* simultaneously on
   the modified friction surface. This gives a new "minutes to nearest service"
   raster comparable to the original travel-time raster T.
4. Compute delta (before − after) and summarize population-level benefits:
   - Total population gaining ≥ N minutes of improvement
   - Population newly within 60/120 minute thresholds
   - Municipality-level breakdown of improvements

Interventions spec
------------------
Interventions are defined in a CSV (or inline defaults):
  - name: label
  - geometry: WKT linestring or reference to shapefile features
  - target_speed_kmh: the post-upgrade speed (e.g., 60 km/h for paved secondary)

If no interventions file exists, the step demonstrates with a synthetic example
that upgrades all flood-risk road cells from their current speed to a target.

Inputs
------
- outputs/rasters/{AOI}_friction_min_per_km.tif    (Step 12)
- PARAMS.TARGET_GRID (original travel-time raster)
- outputs/rasters/{AOI}_pop_1km.tif
- outputs/rasters/{AOI}_roads_flood_risk_cells_1km.tif (Step 04; for default scenario)
- PATHS.SITES (project locations for multi-source cost-distance)

Outputs
-------
- outputs/rasters/{AOI}_sim_travel_after.tif       (new travel surface)
- outputs/rasters/{AOI}_sim_travel_delta.tif       (minutes saved)
- outputs/tables/{AOI}_sim_impact_summary.csv      (population-level delta)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from rasterio.transform import rowcol
from skimage.graph import MCP_Geometric

from config import (
    AOI, PATHS, PARAMS, get_logger, out_r, out_t,
    RESAMPLE,
)
from utils_geo import (
    open_template, write_gtiff, cell_area_km2_latlon,
    align_to_template,
)

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _get_sampling_km(T: xr.DataArray) -> tuple[float, float]:
    """Derive (dy_km, dx_km) from grid at mean latitude."""
    lat_mean = float(np.nanmean(T.y.values))
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.deg2rad(lat_mean))
    tf = T.rio.transform()
    return abs(tf.e) * km_per_deg_lat, abs(tf.a) * km_per_deg_lon


def _multi_source_cost(
    friction: np.ndarray,
    starts: list[tuple[int, int]],
    sampling: tuple[float, float],
    max_cost: float = 300.0,
) -> np.ndarray:
    """
    Compute minimum accumulated cost from *multiple* start points simultaneously.
    Returns a 2-D array of minutes.
    """
    costs = friction.astype("float32").copy()
    costs = np.where(np.isfinite(costs) & (costs > 0), costs, np.float32(np.inf))

    # Filter starts to valid cells
    ny, nx = costs.shape
    valid = [(r, c) for r, c in starts if 0 <= r < ny and 0 <= c < nx and np.isfinite(costs[r, c])]
    if not valid:
        return np.full_like(costs, np.nan)

    mcp = MCP_Geometric(costs, sampling=sampling)
    try:
        res = mcp.find_costs(starts=valid, max_cost=max_cost)
    except TypeError:
        res = mcp.find_costs(valid, max_cost)

    acc = res[0] if isinstance(res, (tuple, list)) else res
    return acc.astype("float32")


def _site_starts(sites: gpd.GeoDataFrame, T: xr.DataArray) -> list[tuple[int, int]]:
    """Convert project site points to (row, col) on the grid."""
    tf = T.rio.transform()
    ny, nx = T.shape
    starts = []
    for _, rec in sites.iterrows():
        geom = rec.geometry
        pt = geom.centroid if geom.geom_type.lower() != "point" else geom
        r, c = rowcol(tf, float(pt.x), float(pt.y))
        if 0 <= r < ny and 0 <= c < nx:
            starts.append((r, c))
    return starts


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    """
    Simulate a road upgrade intervention and quantify population impact.
    """
    # Template & population
    T = open_template(PARAMS.TARGET_GRID)
    ny, nx = T.shape
    tf = T.rio.transform()
    sampling = _get_sampling_km(T)

    pop_fp = PATHS.OUT_R / f"{AOI}_pop_1km.tif"
    if not pop_fp.exists():
        log.error("Population raster not found: %s", pop_fp.name)
        return
    pop = rxr.open_rasterio(pop_fp, masked=True).squeeze()
    if pop.shape != T.shape:
        pop = pop.rio.reproject_match(T, resampling=RESAMPLE("bilinear"))

    # Friction surface from Step 12
    fric_fp = PATHS.OUT_R / f"{AOI}_friction_min_per_km.tif"
    if not fric_fp.exists():
        log.error("Friction raster not found: %s (run Step 12 first)", fric_fp.name)
        return
    friction = rxr.open_rasterio(fric_fp, masked=True).squeeze()
    if friction.shape != T.shape:
        friction = friction.rio.reproject_match(T, resampling=RESAMPLE("bilinear"))

    fric_arr = friction.values.astype("float32").copy()

    # Project sites
    sites_fp = PATHS.SITES
    if not Path(sites_fp).exists():
        log.error("Project sites not found: %s", sites_fp)
        return
    sites = gpd.read_file(sites_fp)
    if sites.crs is None:
        sites.set_crs("EPSG:4326", inplace=True)
    else:
        sites = sites.to_crs("EPSG:4326")
    sites = sites[~sites.geometry.is_empty & sites.geometry.notnull()].copy()

    starts = _site_starts(sites, T)
    if not starts:
        log.error("No valid project site locations on the grid.")
        return
    log.info("Sites: %d project locations as cost-distance sources", len(starts))

    max_cost = float(getattr(PARAMS, "MAX_COST_MIN", 300.0))

    # --- Baseline: multi-source cost-distance on original friction ----------
    log.info("Computing baseline multi-source travel surface...")
    baseline = _multi_source_cost(fric_arr, starts, sampling, max_cost)

    # --- Intervention scenario: upgrade flood-risk road cells ---------------
    # Default scenario: reduce friction on road-risk cells to simulate
    # upgrading vulnerable road segments to secondary-road speed (45 km/h).
    risk_fp = PATHS.OUT_R / f"{AOI}_roads_flood_risk_cells_1km.tif"
    if not Path(risk_fp).exists():
        log.warning("Road-risk raster not found; using synthetic scenario (10%% friction reduction everywhere).")
        fric_after = fric_arr * 0.90  # 10% speed improvement everywhere
        scenario_name = "global_10pct_improvement"
    else:
        risk = rxr.open_rasterio(risk_fp, masked=True).squeeze()
        if risk.shape != T.shape:
            risk = risk.rio.reproject_match(T, resampling=RESAMPLE("nearest"))
        risk_mask = np.nan_to_num(risk.values, nan=0.0) > 0.5

        target_speed_kmh = 45.0  # secondary road standard
        target_friction = 60.0 / target_speed_kmh  # min/km

        fric_after = fric_arr.copy()
        # Only upgrade cells where current friction is worse (slower) than target
        upgrade_mask = risk_mask & np.isfinite(fric_arr) & (fric_arr > target_friction)
        fric_after[upgrade_mask] = target_friction
        n_upgraded = int(np.sum(upgrade_mask))
        log.info("Intervention: upgraded %d flood-risk road cells to %.0f km/h", n_upgraded, target_speed_kmh)
        scenario_name = f"upgrade_risk_roads_to_{int(target_speed_kmh)}kmh"

    # --- After: multi-source cost-distance on modified friction -------------
    log.info("Computing post-intervention travel surface...")
    after = _multi_source_cost(fric_after, starts, sampling, max_cost)

    # --- Delta analysis -----------------------------------------------------
    delta = baseline - after  # positive = time saved (improvement)
    delta = np.where(np.isfinite(delta), delta, 0.0)

    # Write rasters
    after_da = xr.DataArray(after, coords=T.coords, dims=T.dims)
    delta_da = xr.DataArray(delta.astype("float32"), coords=T.coords, dims=T.dims)

    write_gtiff(after_da, out_r("sim_travel_after"), like=T, nodata=np.nan)
    write_gtiff(delta_da, out_r("sim_travel_delta"), like=T, nodata=0)
    log.info("Wrote sim_travel_after.tif and sim_travel_delta.tif")

    # --- Population impact summary ------------------------------------------
    pop_arr = np.nan_to_num(pop.values, nan=0.0)
    area = cell_area_km2_latlon(T).values

    thresholds = [5, 10, 15, 30, 60]  # minutes of improvement
    rows = []
    for thr in thresholds:
        mask = delta >= thr
        rows.append({
            "scenario": scenario_name,
            "min_saved_threshold": thr,
            "pop_gaining": float(np.sum(pop_arr[mask])),
            "area_km2": float(np.sum(area[mask])),
        })

    # New accessibility gains: population crossing the 60/120 min threshold
    for iso in (60, 120):
        before_served = (baseline <= iso) & np.isfinite(baseline)
        after_served = (after <= iso) & np.isfinite(after)
        newly_served = after_served & ~before_served
        rows.append({
            "scenario": scenario_name,
            "min_saved_threshold": f"newly_within_{iso}min",
            "pop_gaining": float(np.sum(pop_arr[newly_served])),
            "area_km2": float(np.sum(area[newly_served])),
        })

    df = pd.DataFrame(rows)
    out_csv = out_t("sim_impact_summary")
    df.to_csv(out_csv, index=False)
    log.info("Wrote %s | scenario=%s", Path(out_csv).name, scenario_name)

    # Quick summary log
    pop_any = float(np.sum(pop_arr[delta > 0]))
    mean_save = float(np.mean(delta[delta > 0])) if np.any(delta > 0) else 0.0
    log.info(
        "Impact: %.0f people gain travel time | avg saving = %.1f min | max saving = %.1f min",
        pop_any, mean_save, float(np.max(delta)),
    )

    log.info("Step 16 complete.")


if __name__ == "__main__":
    main()

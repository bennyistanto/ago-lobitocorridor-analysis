"""
Step 12 — Travel-Time Catchments (road-aware cost-distance)

Purpose
-------
Compute per-project isochrone catchments (e.g., 30/60/120 minutes) using a
road-aware friction raster (minutes per km). This provides more realistic
beneficiary catchments than circular buffers (Step 08), while staying fully
raster-based and AOI-agnostic.

High-level approach
-------------------
1) Build a friction raster (minutes per km):
   - Start with an off-road baseline speed (km/h) → minutes per km.
   - Rasterize OSM roads by class; assign class-specific speeds (km/h).
   - At each cell: pick the *fastest* available speed (road vs off-road) and
     convert to minutes/km (cost rate).

2) Cost-distance from each project point:
   - Use skimage.graph.MCP_Geometric with real-world sampling (km/cell)
     so accumulated cost is in minutes.
   - For each project, compute accumulated minutes up to a max cutoff
     (e.g., 180) and output isochrone masks (30/60/120).

3) (Optional) KPI extraction for each isochrone (population, cropland km², etc.)
   reusing rasters produced in earlier steps.

Inputs (from previous steps / config)
------------------------------------
- PATHS.ROADS (OSM lines; CRS=WGS84) — required
- PATHS.SITES (project points; CRS=WGS84) — required
- PARAMS.TARGET_GRID (1-km template grid to match) — required
- outputs/rasters/{AOI}_pop_1km.tif, {AOI}_cropland_fraction_1km.tif (optional for KPIs)

Config knobs (in config.py; safe defaults used if absent)
--------------------------------------------------------
- ROAD_CLASS_FILTER: tuple[str,...] or ("ALL",) to use all classes
- ROAD_SPEEDS_KMH: dict mapping OSM 'fclass' → km/h (fallback is SPEED_OFFROAD_KMH)
- SPEED_OFFROAD_KMH: float (default ~ 3–5 km/h)
- ISO_THRESH: tuple[int,...] minutes, e.g., (30, 60, 120)
- MAX_COST_MIN: safety cap for cost propagation (default 180)

Outputs
-------
- outputs/rasters/{AOI}_catch_site{N}_{thresh}min.tif   (binary; 1=reached)
- outputs/tables/{AOI}_catch_site_kpis.csv              (optional KPI summary)

Notes / limits
--------------
- Uses a single, isotropic (km/cell) sampling derived from the grid at mean latitude.
  This is usually adequate at 1-km cells; if you need per-row sampling we can extend.
- If 'surface' (unpaved) exists in OSM, you can add a penalty factor in ROAD_SPEEDS_KMH
  or via a SURFACE_MULTIPLIERS dict; not included by default to keep it simple.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import rowcol

import rioxarray as rxr
from skimage.graph import MCP_Geometric

from config import (
    AOI, PATHS, PARAMS, get_logger, out_r, 
    CATCHMENTS_KPI_CSV,
)
from utils_geo import open_template, write_gtiff, cell_area_km2_latlon, align_to_template

log = get_logger(__name__)

# ----------------------------- helpers ---------------------------------------

def _open_align(path: Path, T: xr.DataArray, resampling: str = "nearest") -> xr.DataArray | None:
    """Open raster and align to template T (or return None if missing)."""
    if not Path(path).exists():
        return None
    return align_to_template(path, T, resampling=resampling)


def _get_sampling_km(T: xr.DataArray) -> Tuple[float, float]:
    """
    Derive (dy_km, dx_km) sampling for MCP_Geometric from the grid at mean latitude.
    """
    xs = T.x.values
    ys = T.y.values
    lat_mean = float(np.nanmean(ys))
    # Kilometers per degree (approx; adequate at 1-km resolution)
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.deg2rad(lat_mean))
    # Pixel size in degrees
    tf = T.rio.transform()
    dx_deg = abs(tf.a)
    dy_deg = abs(tf.e)
    return dy_deg * km_per_deg_lat, dx_deg * km_per_deg_lon


def _burn_roads_speed(T: xr.DataArray, roads: gpd.GeoDataFrame, speeds_kmh: Dict[str, float],
                      class_filter: Tuple[str, ...]) -> xr.DataArray:
    """
    Rasterize roads by class → per-cell *max* speed (km/h).
    - roads must be in EPSG:4326
    - class column name is assumed 'fclass' (OSM default you've been using)
    """
    # Optional filter
    if "ALL" not in class_filter:
        roads = roads[roads["fclass"].isin(class_filter)].copy()
    if roads.empty:
        log.warning("No roads after filtering; friction will be off-road only.")
        return xr.full_like(T, np.nan, dtype="float32")

    # For each class, burn a speed surface and keep pixelwise max
    speed_da = xr.full_like(T, np.nan, dtype="float32")
    for cls, spd in speeds_kmh.items():
        sub = roads[roads["fclass"] == cls]
        if sub.empty:
            continue
        try:
            # burn 1 where road of this class exists (using rasterio.features.rasterize)
            shapes = [(geom, 1) for geom in sub.geometry if geom is not None and not geom.is_empty]
            if not shapes:
                continue
            burn_arr = rasterize(
                shapes=shapes,
                out_shape=T.rio.shape,
                transform=T.rio.transform(),
                fill=0,
                dtype="uint8",
                all_touched=True,
            )
            # update speed where burned; keep the fastest per pixel
            arr = speed_da.values.copy()
            mask = (burn_arr == 1)
            # where mask, set to max(existing, spd)
            arr = np.where(mask, np.fmax(np.nan_to_num(arr, nan=0.0), float(spd)), arr)
            speed_da = xr.DataArray(arr.astype("float32"), coords=T.coords, dims=T.dims)
        except Exception as e:
            log.warning(f"Failed rasterizing class '{cls}': {e}")

    # Convert zeros back to NaN (cells without any road class)
    speed_da = speed_da.where(np.isfinite(speed_da) & (speed_da > 0), np.nan)
    return speed_da


def _build_friction_minutes_per_km(T: xr.DataArray, roads_fp: Path) -> xr.DataArray:
    """
    Build friction raster (minutes per km).
    - Off-road baseline: SPEED_OFFROAD_KMH
    - On road cells: take max(speed_offroad, road_speed_class)
    Then friction = 60.0 / speed_kmh  (minutes per km).
    """
    # Load OSM roads
    if not roads_fp.exists():
        raise FileNotFoundError(f"OSM roads not found: {roads_fp}")

    roads = gpd.read_file(roads_fp)
    if roads.crs is None:
        roads.set_crs("EPSG:4326", inplace=True)
    else:
        roads = roads.to_crs("EPSG:4326")

    # Speeds
    try:
        road_speeds = dict(getattr(PARAMS, "ROAD_SPEEDS_KMH"))
    except Exception:
        road_speeds = {
            "motorway": 90, "trunk": 80, "primary": 60, "secondary": 45,
            "tertiary": 35, "unclassified": 30, "residential": 25,
            "track": 20, "service": 20, "path": 5,
        }
    try:
        class_filter = tuple(getattr(PARAMS, "ROAD_CLASS_FILTER"))
    except Exception:
        class_filter = ("motorway","trunk","primary","secondary","tertiary","unclassified","residential","track","service","path")

    off_kmh = float(getattr(PARAMS, "SPEED_OFFROAD_KMH", 4.0))

    # Burn road speeds
    road_speed_da = _burn_roads_speed(T, roads, road_speeds, class_filter)

    # Combine with off-road baseline (keep faster)
    speed = xr.full_like(T, off_kmh, dtype="float32")
    if road_speed_da is not None:
        max_speed = np.fmax(speed.values, np.nan_to_num(road_speed_da.values, nan=0.0))
        speed = xr.DataArray(max_speed.astype("float32"), coords=T.coords, dims=T.dims)

    # Convert to minutes per km
    friction_arr = 60.0 / np.clip(speed.values, 0.1, None)
    friction = xr.DataArray(
        friction_arr.astype("float32"),
        coords=T.coords, dims=T.dims, name="minutes_per_km"
    )
    return friction


def _accumulated_minutes_from_point(friction_min_per_km: xr.DataArray,
                                    start_lon: float, start_lat: float,
                                    max_cost_min: float) -> xr.DataArray:
    """
    Compute accumulated minutes from (lon,lat) using MCP_Geometric.
    We pass 'sampling' so the step length is in kilometers; with costs in min/km,
    the accumulated result is in minutes.
    """
    T = friction_min_per_km
    dy_km, dx_km = _get_sampling_km(T)

    # skimage MCP expects a 2D costs array (float32).
    # Treat NaN/<=0 as impassable by setting to +inf.
    costs = friction_min_per_km.values.astype("float32")
    costs = np.where(np.isfinite(costs) & (costs > 0), costs, np.float32(np.inf))

    # start index in row/col (use rasterio.transform.rowcol for compatibility)
    r, c = rowcol(T.rio.transform(), start_lon, start_lat)
    if (r < 0) or (c < 0) or (r >= T.shape[0]) or (c >= T.shape[1]):
        return xr.full_like(T, np.nan, dtype="float32")
    if not np.isfinite(costs[r, c]):
        # start falls on a barrier/unreachable cell
        return xr.full_like(T, np.nan, dtype="float32")

    mcp = MCP_Geometric(costs, sampling=(dy_km, dx_km))

    # Version-safe call: some skimage versions don’t support `endpoints`,
    # and some return a single array vs (costs, traceback).
    try:
        res = mcp.find_costs(starts=[(r, c)], max_cost=max_cost_min)
    except TypeError:
        # Older signature: positional args (starts, max_cost)
        res = mcp.find_costs([(r, c)], max_cost_min)

    costs_arr = res[0] if isinstance(res, (tuple, list)) else res
    acc = xr.DataArray(costs_arr.astype("float32"), coords=T.coords, dims=T.dims)
    acc = acc.where(np.isfinite(acc))  # unreachable stays NaN
    return acc


def _iso_masks_from_surface(acc_min: xr.DataArray, thresholds: Iterable[int]) -> Dict[int, xr.DataArray]:
    """Build binary masks (1/0) for each minute threshold."""
    out = {}
    for t in thresholds:
        out[t] = (acc_min <= float(t)).fillna(0).astype(np.uint8)
    return out


def _optional_kpis(mask: xr.DataArray, rasters: Dict[str, xr.DataArray], T: xr.DataArray) -> Dict[str, float]:
    area = cell_area_km2_latlon(T)
    m = (mask.values > 0)

    out = {
        "area_km2": float(np.nansum(area.values[m])),
        "pop": np.nan,
        "cropland_km2": np.nan,
        "rwi_mean": np.nan,
        "rwi_pop_weighted": np.nan,
    }

    pop   = rasters.get("pop")
    cropf = rasters.get("cropf")
    rwi   = rasters.get("rwi")

    if pop is not None:
        out["pop"] = float(np.nansum(pop.values[m]))

    if cropf is not None:
        out["cropland_km2"] = float(np.nansum((cropf.values * area.values)[m]))

    if rwi is not None:
        rwi_vals = rwi.values
        # simple (unweighted) mean over reached cells
        rwi_masked = rwi_vals[m]
        if rwi_masked.size > 0:
            out["rwi_mean"] = float(np.nanmean(rwi_masked))

        # population-weighted mean (if pop present and >0)
        if pop is not None:
            pop_vals = pop.values[m]
            w = np.nan_to_num(pop_vals, nan=0.0)
            if np.nansum(w) > 0:
                out["rwi_pop_weighted"] = float(np.nansum(rwi_masked * w) / np.nansum(w))

    return out


# --------------------------------- main --------------------------------------

def main() -> None:
    """
    Build a road-aware friction raster (minutes/km), run cost-distance from each project site,
    save per-threshold isochrone masks, and (optionally) write a KPI table.
    """
    # Template (1-km) grid
    T = open_template(PARAMS.TARGET_GRID)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(f"Template grid | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}")

    # Projects
    sites_fp = PATHS.SITES
    if not Path(sites_fp).exists():
        log.error(f"Project points not found: {sites_fp}")
        return
    sites = gpd.read_file(sites_fp)
    if sites.crs is None:
        sites.set_crs("EPSG:4326", inplace=True)
    else:
        sites = sites.to_crs("EPSG:4326")
    sites = sites[~sites.geometry.is_empty & sites.geometry.notnull()].copy()
    if sites.empty:
        log.warning("No project points found in SITES; nothing to compute.")
        return

    # Friction surface
    log.info("Building friction raster (minutes per km) from OSM roads + off-road baseline...")
    friction = _build_friction_minutes_per_km(T, PATHS.ROADS)
    write_gtiff(friction, PATHS.OUT_R / f"{AOI}_friction_min_per_km.tif", like=T, nodata=np.nan)
    log.info(f"Wrote {AOI}_friction_min_per_km.tif")

    # Optional rasters for KPIs
    pop   = _open_align(PATHS.OUT_R / f"{AOI}_pop_1km.tif", T, "bilinear")
    cropf = _open_align(PATHS.OUT_R / f"{AOI}_cropland_fraction_1km.tif", T, "bilinear")
    rwi   = _open_align(out_r("rwi_meta_1km"), T, "bilinear")
    ras_for_kpis = {"pop": pop, "cropf": cropf, "rwi": rwi}

    thresholds = tuple(getattr(PARAMS, "ISO_THRESH", (30, 60, 120)))
    max_cost_min = float(getattr(PARAMS, "MAX_COST_MIN", max(thresholds) + 60.0))

    # Output KPI rows
    kpi_rows: List[Dict] = []

    # Iterate projects
    for idx, rec in sites.reset_index(drop=True).iterrows():
        geom = rec.geometry
        pt = geom.centroid if geom.geom_type.lower() != "point" else geom
        lon, lat = float(pt.x), float(pt.y)

        # Accumulated minutes
        acc = _accumulated_minutes_from_point(friction, lon, lat, max_cost_min=max_cost_min)

        # Isochrone masks & KPIs
        masks = _iso_masks_from_surface(acc, thresholds)
        for tmin, msk in masks.items():
            out_fp = PATHS.OUT_R / f"{AOI}_catch_site{idx+1}_{tmin}min.tif"
            write_gtiff(msk, out_fp, like=T, nodata=0)

            kpis = _optional_kpis(msk, ras_for_kpis, T)
            row = {
                "site_index": idx+1,
                "lon": lon, "lat": lat,
                "thresh_min": tmin,
                **kpis
            }
            # Mean travel minutes within this isochrone (accumulated cost surface)
            row["mean_travel_min"] = float(np.nanmean(acc.values[msk.values > 0]))

            kpi_rows.append(row)

        log.info(f"Computed isochrones for site {idx+1} at {lon:.5f},{lat:.5f}")

    # Save KPI table
    if kpi_rows:
        df = pd.DataFrame(kpi_rows)
        # --- Enforce schema/dtypes so downstream checks are clean and predictable
        num_cols = ["site_index","lon","lat","thresh_min","area_km2","pop","cropland_km2","rwi_mean","rwi_pop_weighted","mean_travel_min"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # --- Optional: monotonic area check per site (30→60→120 mins non-decreasing)
        if {"site_index","thresh_min","area_km2"}.issubset(df.columns):
            ck = df.sort_values(["site_index","thresh_min"])
            area_nondec = (
                ck.groupby("site_index", group_keys=False)["area_km2"]
                .apply(lambda s: (s.diff().fillna(0) >= -1e-6).all())
            )
            log.info("Catchments KPI: area monotone by site = %.0f%%", 100.0 * area_nondec.mean())

        out_csv = Path(CATCHMENTS_KPI_CSV)
        df.to_csv(out_csv, index=False)
        log.info(f"Wrote catchment KPIs → {out_csv.name} | rows={len(df)}")


    log.info("Step 12 complete.")


if __name__ == "__main__":
    main()

"""
Step 08 — Project KPIs (radius-based catchments)

Purpose
-------
Produce a policy-facing KPI table for last-mile projects (DM sites, Gov/WB/Other),
summarizing beneficiaries and risks in simple radial catchments around each site.

Why radius (for now)?
- Simple, transparent, AOI-agnostic, and runs with only rasters + points.
- Easy to explain to counterparts; no routing engine required.
- Later we can swap/augment with true travel-time catchments (cost-distance).

KPIs per project (computed for each radius in config, e.g., 5/10/30 km)
-----------------------------------------------------------------------
- Population beneficiaries
- Households beneficiaries (= pop / PERSONS_PER_HH)
- Poorest households beneficiaries (uses municipal poverty raster if present)
- Cropland area (km²) = cropland_fraction × cell_area_km2
- % Electrified (share of cells with grid)
- % Rural (share of cells classified rural)
- Priority overlap (share of cells in Top-X mask from Step 07)
- Flood-road risk nearby (count of risky road cells in catchment if present)

Inputs (expected to exist if earlier steps ran)
-----------------------------------------------
Template/rasters:
- PARAMS.TARGET_GRID                     (1-km travel template, defines grid)
- outputs/rasters/{AOI}_pop_1km.tif
- outputs/rasters/{AOI}_cropland_fraction_1km.tif
- outputs/rasters/{AOI}_elec_grid_1km.tif
- outputs/rasters/{AOI}_rural_1km.tif
- outputs/rasters/{AOI}_priority_top10_mask.tif     (from Step 07)
- outputs/rasters/{AOI}_roads_flood_risk_cells_1km.tif (optional, Step 04)
- outputs/rasters/{AOI}_muni_poverty_poverty_rural_1km.tif (optional, Step 06)

Projects:
- PATHS.SITES      (DM sites, may be empty)
- PROJECTS_GOV     (optional)
- PROJECTS_WB      (optional)
- PROJECTS_OTH     (optional)

Config knobs (config.py)
------------------------
- PARAMS.PERSONS_PER_HH
- Optional: CAALA_LP_POINT (lon,lat) — we include distance-to-LP if provided
- Catchment radii (km): set here in this module as CATCHMENT_KM = (5, 10, 30)
  (move to config later if you want)

Outputs
-------
- outputs/tables/{AOI}_project_kpis.csv

Notes / Design Choices
----------------------
- Distances computed in km from (lon,lat) using vectorized haversine.
- Area-true calculations use per-row km² from latitude (cell_area_km2_latlon).
- If a raster is missing (e.g., poverty_rural), KPI columns fall back to NaN with a log.
- If a project layer has 0 rows, we proceed and write an empty table with headers.

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from rasterio.enums import Resampling

from config import (
    AOI, PARAMS, get_logger, out_t,
    PATHS as _P,
    log_denominators,
    RESAMPLE_DEFAULT_CONT, RESAMPLE_DEFAULT_CAT,
)
from utils_geo import open_template, cell_area_km2_latlon

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Tunable: catchment radii (km). You can move this into config later.
CATCHMENT_KM: Tuple[int, ...] = (5, 10, 30)
# ---------------------------------------------------------------------


# ------------------------- small utilities ---------------------------

def _haversine_km(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Vectorized haversine distance (km) between a point (lon1,lat1) and arrays (lon2,lat2).
    lon/lat in degrees.
    """
    R = 6371.0088  # mean Earth radius in km
    lon1r = np.deg2rad(lon1)
    lat1r = np.deg2rad(lat1)
    lon2r = np.deg2rad(lon2)
    lat2r = np.deg2rad(lat2)
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c


def _open_optional(path: Path) -> xr.DataArray | None:
    """Open a raster if it exists; return None otherwise."""
    if not Path(path).exists():
        return None
    import rioxarray as rxr
    return rxr.open_rasterio(path, masked=True).squeeze()


def _load_projects() -> gpd.GeoDataFrame:
    """
    Load available project point layers (SITES + GOV/WB/OTH if present).
    Adds 'source' column to indicate origin layer.
    """
    layers = []
    def _read(fp: Path, tag: str):
        if fp and Path(fp).exists():
            try:
                g = gpd.read_file(fp)
                if not g.empty:
                    g["source"] = tag
                    layers.append(g)
                    log.info(f"Loaded {len(g)} project(s) from {Path(fp).name}")
            except Exception as e:
                log.warning(f"Failed to read {fp}: {e}")

    # DM sites (your existing SITES layer)
    _read(_P.SITES, "dm")

    # Optional others
    try:
        from config import PROJECTS_GOV, PROJECTS_WB, PROJECTS_OTH
        _read(PROJECTS_GOV, "gov")
        _read(PROJECTS_WB,  "wb")
        _read(PROJECTS_OTH, "oth")
    except Exception:
        pass

    if not layers:
        log.warning("No project points found (SITES/GOV/WB/OTH).")
        return gpd.GeoDataFrame(columns=["geometry", "source"], crs="EPSG:4326")

    gdf = pd.concat(layers, ignore_index=True)
    # Ensure WGS84 for distance calc
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _grid_lonlat(T: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return 2D arrays of longitudes and latitudes (center per cell) from template grid.
    Assumes T is in geographic CRS (EPSG:4326).
    """
    xs = T.x.values
    ys = T.y.values
    lon2d, lat2d = np.meshgrid(xs, ys)
    return lon2d, lat2d


def _mask_radius_km(T: xr.DataArray, lon_pt: float, lat_pt: float, radius_km: float,
                    lon2d: np.ndarray, lat2d: np.ndarray) -> xr.DataArray:
    """
    Boolean mask: 1 where cell center is within radius_km of (lon_pt,lat_pt).
    """
    d = _haversine_km(lon_pt, lat_pt, lon2d, lat2d)
    m = (d <= radius_km).astype(np.uint8)
    return xr.DataArray(m, coords=T.coords, dims=T.dims)


# --------------------------- main process ----------------------------

def main() -> None:
    """
    Compute per-project KPIs in radius catchments and write a tidy CSV.
    """
    # Template/CRS/shape
    T = open_template(PARAMS.TARGET_GRID)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(f"Template grid | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}")

    # Rasters (some optional)
    import rioxarray as rxr
    pop   = _open_optional(_P.OUT_R / f"{AOI}_pop_1km.tif")
    cropf = _open_optional(_P.OUT_R / f"{AOI}_cropland_fraction_1km.tif")
    grid  = _open_optional(_P.OUT_R / f"{AOI}_elec_grid_1km.tif")
    rural = _open_optional(_P.OUT_R / f"{AOI}_rural_1km.tif")
    prio  = _open_optional(_P.OUT_R / f"{AOI}_priority_top10_mask.tif")
    risk  = _open_optional(_P.OUT_R / f"{AOI}_roads_flood_risk_cells_1km.tif")  # may be None
    pov   = _open_optional(_P.OUT_R / f"{AOI}_muni_poverty_poverty_rural_1km.tif")  # may be None

    # Ensure all open rasters align to T if they exist
    rasters = {"pop": pop, "cropf": cropf, "grid": grid, "rural": rural, "prio": prio, "risk": risk, "pov": pov}
    for k, da in rasters.items():
        if da is None:
            continue
        if (da.shape != T.shape) or (da.rio.transform() != T.rio.transform()) or (da.rio.crs != T.rio.crs):
            # nearest for masks, bilinear for continuous
            # centralized resampling policy from config
            rs = RESAMPLE_DEFAULT_CONT if k in ("pop", "cropf", "pov") else RESAMPLE_DEFAULT_CAT
            rasters[k] = da.rio.reproject_match(T, resampling=rs)

            log.info(f"Reprojected {k} to match grid")

    pop, cropf, grid, rural, prio, risk, pov = [rasters[k] for k in ["pop","cropf","grid","rural","prio","risk","pov"]]

    # --- AOI-wide denominators snapshot (for consistent logs across steps) ---
    try:
        pop_total = float(np.nansum(pop.values)) if pop is not None else np.nan
        # area-true cropland total (km²)
        area = cell_area_km2_latlon(T)
        crop_total_km2 = float(np.nansum((cropf.values * area.values))) if cropf is not None else np.nan
        # count of electrified grid cells (boolean > 0.5)
        elec_total_cells = int(np.nansum(grid.values > 0.5)) if grid is not None else np.nan
        # optional mean cell area (km²) just for reference in logs
        cell_area_km2 = float(np.nanmean(area.values))
        log_denominators(
            log,
            pop_total=pop_total,
            crop_total_km2=crop_total_km2,
            elec_total_cells=elec_total_cells,
            cell_area_km2=cell_area_km2,
        )
    except Exception as e:
        log.warning(f"Denominator logging skipped: {e}")

    # Required sanity
    if pop is None or cropf is None:
        log.error("Missing required rasters (pop or cropland_fraction). Aborting.")
        return

    # Precompute area (km^2) and lon/lat grids
    area = cell_area_km2_latlon(T)
    lon2d, lat2d = _grid_lonlat(T)

    # Load projects
    projs = _load_projects()
    if projs.empty:
        # Write empty table with headers
        cols = ["project_id","source","lon","lat"]
        for r in CATCHMENT_KM:
            cols += [
                f"pop_{r}km","hh_{r}km","poorest_hh_{r}km","ag_km2_{r}km",
                f"pct_electrified_{r}km","pct_rural_{r}km","pct_priority_{r}km","risk_roadcells_{r}km"
            ]
        df_empty = pd.DataFrame(columns=cols)
        out_csv = out_t("project_kpis")
        df_empty.to_csv(out_csv, index=False)
        log.info(f"No projects found. Wrote empty table → {Path(out_csv).name}")
        return

    # Build result rows
    rows: List[Dict] = []
    persons_per_hh = float(PARAMS.PERSONS_PER_HH or 5.0)

    for i, rec in projs.iterrows():
        geom = rec.geometry
        if geom is None or geom.is_empty:
            continue
        # Point only; if line/polygon, take centroid
        pt = geom.centroid if geom.geom_type.lower() != "point" else geom
        lon, lat = float(pt.x), float(pt.y)

        row = {
            "project_id": rec.get("id", i),
            "source": rec.get("source", "unknown"),
            "lon": lon, "lat": lat,
        }
        # Tag row with AOI for tidy downstream merges
        row["aoi"] = AOI


        # Per-radius metrics
        for rkm in CATCHMENT_KM:
            m = _mask_radius_km(T, lon, lat, rkm, lon2d, lat2d)

            # Handle NaNs safely and establish the denominator (# cells in catchment)
            m_bool = (m.values > 0)
            denom_cells = int(m_bool.sum())
            row[f"denom_cells_{rkm}km"] = denom_cells  # <- explicit denominator saved

            # population
            pop_sum = float(np.nansum(pop.values[m_bool])) if pop is not None else np.nan
            hh_sum  = (pop_sum / persons_per_hh) if np.isfinite(pop_sum) else np.nan

            # poorest households (if pov available: pov is 0..1)
            if pov is not None:
                pov_rate = pov.values
                poorest_hh = float(np.nansum((pop.values / persons_per_hh) * pov_rate * m_bool))
            else:
                poorest_hh = np.nan

            # agricultural area (km²)
            ag_km2 = float(np.nansum((cropf.values * area.values) * m_bool)) if cropf is not None else np.nan

            # % electrified (grid == 1)
            if grid is not None:
                hits  = int(np.nansum((grid.values > 0.5) * m_bool))
                pct_elec = (hits / denom_cells) if denom_cells > 0 else np.nan
            else:
                pct_elec = np.nan

            # % rural
            if rural is not None:
                hits  = int(np.nansum((rural.values > 0.5) * m_bool))
                pct_rural = (hits / denom_cells) if denom_cells > 0 else np.nan
            else:
                pct_rural = np.nan

            # % in priority mask (Step 07)
            if prio is not None:
                hits  = int(np.nansum((prio.values > 0.5) * m_bool))
                pct_prio = (hits / denom_cells) if denom_cells > 0 else np.nan
            else:
                pct_prio = np.nan

            # flood-road risk cells count (Step 04)
            if risk is not None:
                risk_count = int(np.nansum((risk.values > 0.5) * m_bool))
            else:
                risk_count = np.nan

            # write into row
            row.update({
                f"pop_{rkm}km": pop_sum,
                f"hh_{rkm}km": hh_sum,
                f"poorest_hh_{rkm}km": poorest_hh,
                f"ag_km2_{rkm}km": ag_km2,
                f"pct_electrified_{rkm}km": pct_elec,
                f"pct_rural_{rkm}km": pct_rural,
                f"pct_priority_{rkm}km": pct_prio,
                f"risk_roadcells_{rkm}km": risk_count,
            })

        rows.append(row)

    df = pd.DataFrame(rows)

    # Optional: distances to named reference points (if any defined in config)
    try:
        from config import REFERENCE_POINTS
        for rp in (REFERENCE_POINTS or []):
            try:
                name = str(rp["name"]).strip().lower()
                lon0 = float(rp["lon"]); lat0 = float(rp["lat"])
                df[f"dist_km_to_{name}"] = _haversine_km(lon0, lat0, df["lon"].values, df["lat"].values)
            except Exception as e:
                log.warning(f"Skip reference point {rp}: {e}")
    except Exception:
        pass


    # Sort for readability (largest 10km population first)
    if f"pop_10km" in df.columns:
        df.sort_values(by="pop_10km", ascending=False, inplace=True)

    out_csv = out_t("project_kpis")
    df.to_csv(out_csv, index=False)
    log.info(f"Saved project KPIs → {Path(out_csv).name} | rows={len(df)}")

    log.info("Step 08 complete.")


if __name__ == "__main__":
    main()

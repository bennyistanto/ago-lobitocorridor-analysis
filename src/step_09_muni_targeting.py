"""
Step 09 — Municipality Targeting Table (Admin2 shortlist)

Purpose
-------
Create a corridor-ready, sortable shortlist of municipalities (admin2) using:
- RAPP indicators (Step 06 wide table) such as rural poverty & food insecurity
- Raster KPIs computed on the 1-km grid:
    • total area (km²)
    • total population
    • population within ≤60 and ≤120 minutes to market/finance (from travel-time raster)
    • cropland area (km²) using cropland_fraction × cell_area_km2
    • % electrified cells (grid presence)
    • % rural cells
    • % area selected by the tunable priority mask (Step 07)

It outputs a single CSV per AOI with a defensible, transparent **composite score** to help
concentrate last-mile investments. The score is a simple average of min–max normalized
indicators (see "Composite score" below).

Inputs (from previous steps / config)
------------------------------------
- PATHS.MUNI_CLEAN_TBL     : {AOI}_municipality_indicators.csv  (Step 06)
- PARAMS.TARGET_GRID       : travel-time raster (minutes) — used as the 1-km template
- outputs/rasters/{AOI}_pop_1km.tif
- outputs/rasters/{AOI}_cropland_fraction_1km.tif
- outputs/rasters/{AOI}_elec_grid_1km.tif
- outputs/rasters/{AOI}_rural_1km.tif
- outputs/rasters/{AOI}_priority_top10_mask.tif  (Step 07; optional but recommended)

Admin2 geometry source
----------------------
We need admin2 polygons to aggregate rasters. We automatically pick a representative
RAPP shapefile from PATHS.MUNI_DIR — preference order:
    1) poverty theme for this AOI (ago_gov_{aoi}_poverty_rapp_2020_a.shp)
    2) otherwise, the first available theme file for this AOI

Outputs
-------
- outputs/tables/{AOI}_priority_muni_rank.csv

Design choices / notes
----------------------
- No reliance on Step 01 saved isochrones: we derive ≤60/≤120 masks on the fly from travel time.
- Area uses per-row latitude-based km² from utils_geo.cell_area_km2_latlon() (area-true enough for 1-km grids).
- % electrified / % rural / % priority are computed as share of valid grid cells per municipality.
- Composite score (simple, explainable, and AOI-agnostic by default):
    * + poverty_rural (higher → higher priority)
    * + food_insec_scale (if available; higher → higher priority)
    * + avg_travel_time_minutes (higher → higher priority)
    * + pct_priority_area (higher → higher priority)
    * + cropland_km2 (higher → higher priority)
    * + (1 - pct_electrified_cells) (lower electrification → higher priority)
  Each factor is min–max scaled over municipalities; score is the average of available factors.
  You can later move weights to config if you want differential emphasis.

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from rasterio.features import rasterize

from config import (
    AOI, PATHS, PARAMS, get_logger, out_t,
    muni_path_for, muni_glob_for_theme, ADMIN2_THEMES,
)
from utils_geo import cell_area_km2_latlon, write_gtiff_masked  # area used; write_gtiff available if you want to persist labels

log = get_logger(__name__)


# ------------------------------ helpers --------------------------------------

def _pick_admin2_geom() -> Path | None:
    """Pick an admin2 shapefile for this AOI. Prefer poverty theme; else first available."""
    # Try the preferred theme first
    for p in muni_path_for(AOI, "poverty"):  # returns a list of candidate Paths
        if p.exists():
            return p

    # Fallback: first existing file among any configured themes
    for theme in ADMIN2_THEMES:
        for p in muni_path_for(AOI, theme):
            if p.exists():
                return p

    return None


def _admin2_labels_raster(T: xr.DataArray, shp: Path) -> tuple[xr.DataArray, pd.DataFrame]:
    """
    Read an admin2 shapefile and burn integer labels to the 1-km grid.

    Returns
    -------
    labels : xr.DataArray
        2-D (y,x) int array with NaN outside polygons; CRS/transform from T.
    lut : pd.DataFrame
        Lookup table: label -> ADM2CD_c, NAM_1, NAM_2
    """
    # Read polygons
    gdf = gpd.read_file(shp)

    # Make sure we have identifiers
    if "ADM2CD_c" not in gdf.columns:
        # Convert whatever exists to a stable categorical code starting at 1
        codes, uniques = pd.factorize(gdf.index, sort=True)
        gdf["ADM2CD_c"] = (codes + 1).astype(int)

    # Create (or reuse) an integer label column
    burn_field = "ADM2_label"
    if burn_field not in gdf.columns:
        gdf[burn_field] = pd.factorize(gdf["ADM2CD_c"], sort=True)[0] + 1

    # Build the LUT for later joins: ensure the key is ADM2_label (int)
    lut = (
        gdf[[burn_field, "ADM2CD_c"]]
        .rename(columns={burn_field: "ADM2_label"})
        .copy()
    )
    for col in ("NAM_1", "NAM_2"):
        if col in gdf.columns:
            lut[col] = gdf[col].values
    lut["ADM2_label"] = lut["ADM2_label"].astype(int)

    # Ensure geometries are in the same CRS as the template grid
    try:
        target_crs = T.rio.crs
        if gdf.crs is None:
            # Assume EPSG:4326 if missing; adjust if your data differs
            gdf = gdf.set_crs("EPSG:4326")
        if (target_crs is not None) and (gdf.crs != target_crs):
            gdf = gdf.to_crs(target_crs)
    except Exception:
        # Fallback: leave as-is if CRS metadata is problematic
        pass

    # Rasterize with all_touched=True to be inclusive at boundaries
    shapes = ((geom, int(val)) for geom, val in zip(gdf.geometry, gdf[burn_field]))
    arr = rasterize(
        shapes=shapes,
        out_shape=T.rio.shape,
        transform=T.rio.transform(),
        fill=0,
        all_touched=True,
        dtype="int32",
    )

    labels = xr.DataArray(arr, coords={"y": T.y, "x": T.x}, dims=("y", "x")).astype("float32")
    labels = labels.where(labels > 0, np.nan)  # outside polygons -> NaN

    # carry georeferencing
    try:
        labels.rio.write_crs(T.rio.crs, inplace=True)
        labels.rio.write_transform(T.rio.transform(), inplace=True)
    except Exception:
        pass

    return labels, lut


def _open_align(path: Path, T: xr.DataArray, resampling: str | Resampling = "bilinear") -> xr.DataArray | None:
    """Open a raster and reproject to match T if needed; None if missing."""
    if not path.exists():
        return None
    da = rxr.open_rasterio(path, masked=True).squeeze()
    if (da.shape != T.shape) or (da.rio.transform() != T.rio.transform()) or (da.rio.crs != T.rio.crs):
        # Allow either "bilinear"/"nearest" or Resampling enum
        if isinstance(resampling, str):
            try:
                resampling = getattr(Resampling, resampling)
            except AttributeError:
                raise ValueError(f"Unknown resampling='{resampling}'")
        da = da.rio.reproject_match(T, resampling=resampling)
    return da


def _groupby_sum_mean(labels: xr.DataArray, value_da: xr.DataArray, how: str = "sum") -> pd.DataFrame:
    """
    Aggregate a value raster by admin2 labels. Returns DataFrame with columns:
    [ADM2_label, val] where 'val' is sum or mean over label.
    """
    lab = labels.values
    val = value_da.values
    m = np.isfinite(lab) & np.isfinite(val)
    if not np.any(m):
        return pd.DataFrame(columns=["ADM2_label", "val"])
    lab1 = lab[m].astype(np.int64)
    val1 = val[m].astype(np.float64)
    if how == "sum":
        agg = np.bincount(lab1, weights=val1)
    elif how == "mean":
        s = np.bincount(lab1, weights=val1)
        c = np.bincount(lab1)
        with np.errstate(invalid="ignore", divide="ignore"):
            agg = s / c
    else:
        raise ValueError("how must be 'sum' or 'mean'")
    idx = np.arange(len(agg), dtype=int)
    df = pd.DataFrame({"ADM2_label": idx, "val": agg})
    # Drop background (label 0) and align types with LUT
    df = df[df["ADM2_label"] > 0].copy()
    df["ADM2_label"] = df["ADM2_label"].astype(int)
    return df


def _minmax(series: pd.Series) -> pd.Series:
    """
    Min–max scale to 0..1 (ignoring NaNs). Returns zeros if constant/empty.
    Avoids RuntimeWarning from nanmin/nanmax on all-NaN inputs.
    """
    x = pd.to_numeric(series, errors="coerce")
    # Only finite values are considered
    m = np.isfinite(x.values)
    if not m.any():
        # all-NaN → return zeros (same length, aligned index)
        return pd.Series(np.zeros(len(x)), index=x.index)

    xv = x.values[m]
    lo = np.min(xv)
    hi = np.max(xv)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # constant (or invalid bounds) → zeros
        return pd.Series(np.zeros(len(x)), index=x.index)

    out = np.zeros(len(x), dtype="float64")
    out[m] = (xv - lo) / (hi - lo)
    return pd.Series(out, index=x.index)


# ------------------------------- main ----------------------------------------

def main() -> None:
    """
    Build admin2 targeting table by combining RAPP-wide indicators with raster KPIs,
    then write {AOI}_priority_muni_rank.csv sorted by a transparent composite score.
    """
    # Template grid
    T = rxr.open_rasterio(PARAMS.TARGET_GRID, masked=True).squeeze()
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(f"Template grid | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}")

    # Choose admin2 geometry source
    shp = _pick_admin2_geom()
    if shp is None:
        raise FileNotFoundError("No admin2 RAPP shapefile found for this AOI under PATHS.MUNI_DIR.")
    log.info(f"Using admin2 geometry from {shp.name}")

    # Rasterize admin2 labels; get lookup
    labels, lut = _admin2_labels_raster(T, shp)  # labels in ADM2_label (int), NaN outside
    # Optionally persist the label grid for debugging:
    write_gtiff_masked(labels, PATHS.OUT_R / f"{AOI}_admin2_labels_1km.tif", like=T, nodata=np.nan)

    # Open aligned rasters
    rwi = _open_align(PATHS.OUT_R / f"{AOI}_rwi_1km.tif", T, "bilinear")
    pop    = _open_align(PATHS.OUT_R / f"{AOI}_pop_1km.tif", T, "bilinear")
    cropf  = _open_align(PATHS.OUT_R / f"{AOI}_cropland_fraction_1km.tif", T, "bilinear")
    grid   = _open_align(PATHS.OUT_R / f"{AOI}_elec_grid_1km.tif", T, "nearest")
    rural  = _open_align(PATHS.OUT_R / f"{AOI}_rural_1km.tif", T, "nearest")
    prio   = _open_align(PATHS.OUT_R / f"{AOI}_priority_top10_mask.tif", T, "nearest")  # may be None

    if pop is None or cropf is None or rural is None:
        raise RuntimeError("Missing required rasters: pop_1km, cropland_fraction_1km, rural_1km")

    # Area (km²) by cell
    area = cell_area_km2_latlon(T)

    # Travel-time-based masks on the fly (≤60 / ≤120 minutes)
    iso60 = (T <= 60.0).astype(np.uint8)
    iso120 = (T <= 120.0).astype(np.uint8)

    # --- Aggregations by admin2 --------------------------------------
    # area_km2
    area_df = _groupby_sum_mean(labels, area, how="sum").rename(columns={"val": "area_km2"})
    # pop total
    pop_df = _groupby_sum_mean(labels, pop, how="sum").rename(columns={"val": "pop_total"})
    # pop within 60/120 min
    pop60_df = _groupby_sum_mean(labels, (pop * iso60), how="sum").rename(columns={"val": "pop_le60min"})
    pop120_df = _groupby_sum_mean(labels, (pop * iso120), how="sum").rename(columns={"val": "pop_le120min"})
    # cropland area (km²)
    ag_df = _groupby_sum_mean(labels, (cropf * area), how="sum").rename(columns={"val": "cropland_km2"})
    # % electrified cells (grid > 0.5)
    elc_df = _groupby_sum_mean(labels, (grid > 0.5).astype(np.uint8), how="mean").rename(columns={"val": "pct_electrified"})
    # % rural cells
    rur_df = _groupby_sum_mean(labels, (rural > 0.5).astype(np.uint8), how="mean").rename(columns={"val": "pct_rural"})
    # % priority area (if available)
    if prio is not None:
        pr_df = _groupby_sum_mean(labels, (prio > 0.5).astype(np.uint8), how="mean").rename(columns={"val": "pct_priority"})
    else:
        pr_df = pd.DataFrame({"ADM2_label": [], "pct_priority": []})

    # RWI: mean by admin2 if available (Meta RWI -2..+2; lower = poorer)
    if rwi is not None:
        rwi_mean_df = _groupby_sum_mean(labels, rwi, how="mean").rename(columns={"val": "rwi_mean"})
    else:
        rwi_mean_df = pd.DataFrame({"ADM2_label": [], "rwi_mean": []})

    # Merge all stats
    stats = (
        area_df.merge(pop_df, on="ADM2_label", how="outer")
               .merge(pop60_df, on="ADM2_label", how="outer")
               .merge(pop120_df, on="ADM2_label", how="outer")
               .merge(ag_df, on="ADM2_label", how="outer")
               .merge(elc_df, on="ADM2_label", how="outer")
               .merge(rur_df, on="ADM2_label", how="outer")
               .merge(pr_df, on="ADM2_label", how="outer")
               .merge(rwi_mean_df, on="ADM2_label", how="outer")

    )

    # Attach names/codes
    stats = stats.merge(lut, on="ADM2_label", how="left")

    # Bring in RAPP-wide indicators (Step 06) — collapse to one row per ADM2
    if Path(PATHS.MUNI_CLEAN_TBL).exists():
        wide = pd.read_csv(PATHS.MUNI_CLEAN_TBL)

        key_cols = ["ADM2CD_c", "NAM_1", "NAM_2"]
        want_cols = key_cols + [
            c for c in (
                "poverty__poverty_rural",
                "foodinsecurity__food_insec_scale",
                "traveltime__avg_hours_to_market_financial",
            ) if c in wide.columns
        ]

        wide = wide[want_cols].copy()
        # Harmonize key types
        for k in key_cols:
            if k in wide.columns:
                wide[k] = wide[k].astype(str)
        for k in key_cols:
            if k in stats.columns:
                stats[k] = stats[k].astype(str)

        # Collapse to one row per ADM2 (first non-null wins)
        before = len(wide)
        wide = (wide
                .sort_values(key_cols)
                .groupby(key_cols, as_index=False)
                .first())
        after = len(wide)
        if before != after:
            log.info("Collapsed Step 06 table from %d to %d rows (unique ADM2).", before, after)

        # Merge (prefer RAPP names if available)
        stats = stats.merge(wide, on=key_cols, how="left")
    else:
        log.warning("%s not found; proceeding without RAPP attributes.", Path(PATHS.MUNI_CLEAN_TBL).name)

    # Rename headline RAPP columns (if present)
    if "poverty__poverty_rural" in stats.columns:
        stats.rename(columns={"poverty__poverty_rural": "poverty_rural"}, inplace=True)
    if "foodinsecurity__food_insec_scale" in stats.columns:
        stats.rename(columns={"foodinsecurity__food_insec_scale": "food_insec_scale"}, inplace=True)
    if "traveltime__avg_hours_to_market_financial" in stats.columns:
        stats.rename(columns={"traveltime__avg_hours_to_market_financial": "avg_hours_to_market_financial"}, inplace=True)

    # Derived indicators
    stats["pop_gt120min"] = stats["pop_total"] - stats["pop_le120min"]

    # Composite score (min–max across municipalities), transparent defaults
    # Higher priority if: higher poverty, higher food insecurity, higher average travel time,
    # larger priority share, larger cropland area, lower electrification.
    comp_parts: Dict[str, pd.Series] = {}
    if "poverty_rural" in stats.columns:
        comp_parts["poverty"] = _minmax(stats["poverty_rural"])
    if "food_insec_scale" in stats.columns:
        comp_parts["food_insec"] = _minmax(stats["food_insec_scale"])
    # average travel time (from RAPP, hours) → minutes to keep units intuitive; else estimate via pop-weighted TT if desired
    if "avg_hours_to_market_financial" in stats.columns:
        comp_parts["avg_tt"] = _minmax(stats["avg_hours_to_market_financial"])
    if "pct_priority" in stats.columns:
        comp_parts["priority_area"] = _minmax(stats["pct_priority"])
    comp_parts["cropland"] = _minmax(stats["cropland_km2"])
    comp_parts["elec_inverse"] = _minmax(1.0 - stats["pct_electrified"])
    # RWI: lower RWI => poorer => higher priority
    if "rwi_mean" in stats.columns:
        comp_parts["rwi_inverse"] = _minmax(-stats["rwi_mean"])

    # combine (simple mean of available components)
    if comp_parts:
        comp_df = pd.DataFrame(comp_parts)
        stats["score"] = comp_df.mean(axis=1).fillna(0.0)
    else:
        stats["score"] = 0.0

    # Order & clean
    out_cols = [
        "ADM2CD_c", "NAM_1", "NAM_2",
        "area_km2", "pop_total",
        "pop_le60min", "pop_le120min", "pop_gt120min",
        "cropland_km2", "pct_electrified", "pct_rural",
    ]
    if "pct_priority" in stats.columns:
        out_cols.append("pct_priority")
    if "poverty_rural" in stats.columns:
        out_cols.append("poverty_rural")
    if "food_insec_scale" in stats.columns:
        out_cols.append("food_insec_scale")
    if "avg_hours_to_market_financial" in stats.columns:
        out_cols.append("avg_hours_to_market_financial")
    if "rwi_mean" in stats.columns:
        out_cols.append("rwi_mean")

    out_cols.append("score")

    out = stats[out_cols].copy()
    out.sort_values(by="score", ascending=False, inplace=True)
    out.reset_index(drop=True, inplace=True)

    # Save
    out_csv = out_t("priority_muni_rank")
    out.to_csv(out_csv, index=False)
    log.info(f"Saved municipality targeting table → {Path(out_csv).name} | rows={len(out)}")

    log.info("Step 09 complete.")


if __name__ == "__main__":
    main()

"""
Step 11 — Priority clusters & hotspot stats

Purpose
-------
Convert the binary priority mask into coherent *clusters* (8-neighborhood connected
components), prune tiny speckles, and compute per-cluster KPIs the team cares about:
- area (km²), population, cropland km²
- % electrified cells, % rural cells
- avg travel time (minutes), avg drought (0..1 or %), optional flood-road risk cells
- dominant municipality (admin2) by cell count, with names/codes
- cluster centroid (lon, lat) for quick map tips

Inputs (expected on disk; alignment is auto-checked)
----------------------------------------------------
- PARAMS.TARGET_GRID                          (the 1-km template; minutes to market)
- outputs/rasters/{AOI}_priority_top10_mask.tif          (Step 07/10)
- outputs/rasters/{AOI}_pop_1km.tif                      (Step 00)
- outputs/rasters/{AOI}_cropland_fraction_1km.tif        (Step 00)
- outputs/rasters/{AOI}_elec_grid_1km.tif                (Step 00)
- outputs/rasters/{AOI}_rural_1km.tif                    (Step 00)
- outputs/rasters/{AOI}_drought_1km.tif                  (Step 00; optional)
- outputs/rasters/{AOI}_roads_flood_risk_cells_1km.tif   (Step 04; optional)
- outputs/rasters/{AOI}_flood_rp100_maxdepth_1km.tif     (Step 00; optional)

Admin2 geometry (for dominant municipality)
-------------------------------------------
We rasterize admin2 polygons (from your RAPP admin2 files in PATHS.MUNI_DIR) to
the 1-km grid and compute the *mode* label inside each cluster. Preference order:
    1) poverty theme file for this AOI (ago_gov_{aoi}_poverty_rapp_2020_a.shp)
    2) otherwise, the first available theme file for this AOI

Config knobs (local defaults here; move to config.py later if you prefer)
------------------------------------------------------------------------
- MIN_CLUSTER_CELLS = max( int(PARAMS.MIN_CLUSTER_CELLS or 30), 1 )
- Also supports an *area* threshold in km²:
    MIN_CLUSTER_KM2 = 2.0  (set 0 to disable)
- WRITE_CLUSTER_RASTER = True  → writes label raster
- WRITE_PER_CELL_CLUSTER_ID  = False (debug: heavy; writes cluster id per cell)

Outputs
-------
- outputs/rasters/{AOI}_priority_clusters_1km.tif               (label raster: 1..K)
- outputs/tables/{AOI}_priority_clusters.csv                     (cluster-level KPIs)
- (optional) outputs/rasters/{AOI}_priority_clusterid_1km.tif    (same as clusters; kept for compatibility)

Notes / design choices
----------------------
- Connectivity: 8-neighborhood ensures diagonally touching cells belong together.
- Small-object pruning: by *cells* first, then by *km²* (robust to latitude).
- Drought units auto-detected (0..1 vs 0..100).
- Flood-road risk is counted as *cells flagged* within the cluster (simple, clear).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from scipy.ndimage import label

from config import (
    AOI, PATHS, PARAMS, get_logger, out_r, out_t,
    muni_path_for, muni_glob_for_theme,
    ADMIN2_THEMES,
    PRIORITY_TOP10_TIF,
    PRIORITY_CLUSTERS_TIF,
)

from utils_geo import cell_area_km2_latlon, write_gtiff, align_to_template

log = get_logger(__name__)

# ----------------------- local “config” for this step -------------------------

WRITE_CLUSTER_RASTER: bool = True
WRITE_PER_CELL_CLUSTER_ID: bool = False  # debug / heavy

# If PARAMS.MIN_CLUSTER_CELLS is set, we honor it; else default to 30
MIN_CLUSTER_CELLS: int = max(int(getattr(PARAMS, "MIN_CLUSTER_CELLS", 30) or 30), 1)
# Also prune by km² (helps when resolution differs or latitudinal area varies)
MIN_CLUSTER_KM2: float = float(getattr(PARAMS, "MIN_CLUSTER_KM2", 2.0))  # set 0 to disable

# ------------------------------- helpers -------------------------------------

def _open_align(path: Path, T: xr.DataArray, resampling: str) -> xr.DataArray | None:
    """Open a raster and reproject to match T if needed; None if missing."""
    if not Path(path).exists():
        return None
    return align_to_template(path, T, resampling=resampling)


def _pick_admin2_geom() -> Path | None:
    """Pick an Admin2 shapefile for this AOI (prefer 'poverty'; else first existing theme)."""
    # muni_path_for returns a list[Path]; pick the first that exists
    cand = muni_path_for(AOI, "poverty")
    for p in cand:
        if p.exists():
            return p
    for theme in ADMIN2_THEMES:
        for p in muni_path_for(AOI, theme):
            if p.exists():
                return p
    return None


def _admin2_label_grid(T: xr.DataArray) -> Tuple[xr.DataArray, pd.DataFrame]:
    """
    Rasterize admin2 polygons to 1-km labels keyed by ADM2CD_c.
    Returns:
      labels (DataArray, int64, NaN=outside),
      lookup DF: [ADM2_label, ADM2CD_c, NAM_1, NAM_2]
    """
    shp = _pick_admin2_geom()
    if shp is None:
        log.warning("No admin2 shapefile found; clusters will have no dominant municipality.")
        return xr.full_like(T, np.nan, dtype="float32"), pd.DataFrame(columns=["ADM2_label","ADM2CD_c","NAM_1","NAM_2"])

    gdf = gpd.read_file(shp)
    need = ["ADM2CD_c", "NAM_1", "NAM_2", "geometry"]
    miss = [c for c in need if c not in gdf.columns]
    if miss:
        raise ValueError(f"{shp.name} missing columns: {miss}")

    gdf = gdf.to_crs("EPSG:4326") if gdf.crs is not None else gdf.set_crs("EPSG:4326")
    # map to numeric labels 1..N
    codes = gdf["ADM2CD_c"].astype(str)
    labels_series, uniques = pd.factorize(codes)
    gdf["ADM2_label"] = labels_series + 1  # 1..N
    lut = pd.DataFrame({
        "ADM2_label": np.arange(1, len(uniques)+1, dtype=np.int64),
        "ADM2CD_c": uniques,
    }).merge(gdf[["ADM2_label","NAM_1","NAM_2"]].drop_duplicates(), on="ADM2_label", how="left")

    # rasterize (all_touched=True for crisp boundaries)
    # Use rioxarray's rasterize via GeoSeries accessor for simplicity:
    labels = gdf[["ADM2_label", "geometry"]].rio.write_crs("EPSG:4326", inplace=True).rasterize(
        out_shape=T.rio.shape, transform=T.rio.transform(), all_touched=True, fill=0, dtype=np.int64
    )
    labels = labels.where(labels > 0, np.nan)
    labels.rio.write_crs(T.rio.crs, inplace=True)
    labels.rio.write_transform(T.rio.transform(), inplace=True)
    return labels, lut


def _connected_components(mask: xr.DataArray) -> xr.DataArray:
    """Label 8-neighborhood connected components in a binary mask; return int32 labels 0..K (0=background)."""
    arr = (mask.values > 0).astype(np.uint8)
    structure = np.ones((3,3), dtype=np.uint8)  # 8-neigh
    lbl, n = label(arr, structure=structure)
    log.info(f"Connected components found: {n}")
    return xr.DataArray(lbl.astype(np.int32), coords=mask.coords, dims=mask.dims)


def _prune_clusters(lbl_da: xr.DataArray, T: xr.DataArray, min_cells: int, min_km2: float) -> xr.DataArray:
    """Set labels of small clusters to 0 if they are < min_cells or < min_km2."""
    lbl = lbl_da.values
    nmax = lbl.max()
    if nmax <= 0:
        return lbl_da

    area = cell_area_km2_latlon(T).values
    keep = np.ones(nmax+1, dtype=np.uint8)  # index 0 unused (background)

    # counts and area per label
    counts = np.bincount(lbl.ravel())
    # km² per label
    area_sum = np.bincount(lbl.ravel(), weights=area.ravel())

    for k in range(1, nmax+1):
        if counts[k] < min_cells:
            keep[k] = 0
        elif (min_km2 > 0.0) and (area_sum[k] < min_km2):
            keep[k] = 0

    pruned = lbl.copy()
    kill_ids = np.where(keep == 0)[0]
    for k in kill_ids:
        pruned[lbl == k] = 0

    kept = int((keep[1:] > 0).sum())
    log.info(f"Pruned clusters: kept={kept} | removed={(len(keep)-1 - kept)} by cells<{min_cells} or km2<{min_km2}")
    return xr.DataArray(pruned.astype(np.int32), coords=lbl_da.coords, dims=lbl_da.dims)


def _cluster_stats(lbl_da: xr.DataArray,
                   T: xr.DataArray,
                   rasters: Dict[str, xr.DataArray],
                   admin2_lbl: xr.DataArray | None,
                   admin2_lut: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compute per-cluster stats by fast bincounts.
    Returns tidy DataFrame with one row per cluster_id (>=1).
    """
    lbl = lbl_da.values
    ids = np.unique(lbl)
    ids = ids[ids > 0]
    if ids.size == 0:
        return pd.DataFrame(columns=["cluster_id"])

    area = cell_area_km2_latlon(T).values
    # helper to sum by label
    def sum_by_label(values: np.ndarray) -> np.ndarray:
        return np.bincount(lbl.ravel(), weights=values.ravel(), minlength=lbl.max()+1)

    def mean_by_label(values: np.ndarray) -> np.ndarray:
        s = np.bincount(lbl.ravel(), weights=values.ravel(), minlength=lbl.max()+1)
        c = np.bincount(lbl.ravel(), minlength=lbl.max()+1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / np.maximum(c, 1)
        return m

    # base rasters
    pop   = rasters.get("pop")
    cropf = rasters.get("cropf")
    grid  = rasters.get("grid")
    rural = rasters.get("rural")
    drt   = rasters.get("drt")
    risk  = rasters.get("risk")
    flood = rasters.get("flood")
    tt    = T

    # sums / means per label (index = label id)
    area_km2 = sum_by_label(area)
    pop_sum  = sum_by_label(pop.values) if pop is not None else None
    ag_km2   = sum_by_label((cropf.values * area)) if cropf is not None else None

    pct_elec = mean_by_label((grid.values > 0.5).astype(np.uint8)) if grid is not None else None
    pct_rur  = mean_by_label((rural.values > 0.5).astype(np.uint8)) if rural is not None else None
    tt_mean  = mean_by_label(tt.values)  # minutes
    # drought: handle 0..100 vs 0..1 by median heuristic, then average
    dr_mean = None
    if drt is not None:
        med = float(np.nanmedian(drt.values))
        dd = drt.values/100.0 if med > 1.0 else drt.values
        dr_mean = mean_by_label(dd)

    # Relative Wealth Index (RWI) mean (−2..2 typically, but we don't rescale here)
    rwi_mean = None
    if rasters.get("rwi") is not None:
        rwi_mean = mean_by_label(rasters["rwi"].values.astype("float32"))

    # flood-road risk (count flagged cells)
    risk_cnt = sum_by_label((risk.values > 0.5).astype(np.uint8)) if risk is not None else None
    # flood depth (mean)
    flood_mean = mean_by_label(flood.values) if flood is not None else None

    # dominant municipality (mode of admin2 label)
    dom_admin = None
    if admin2_lbl is not None and admin2_lut is not None and not admin2_lut.empty:
        a = admin2_lbl.values
        # for each cluster id, compute the most frequent admin2_label
        dom_ids = np.zeros(lbl.max()+1, dtype=np.float32)
        for k in ids:
            mask = (lbl == k) & np.isfinite(a)
            if not np.any(mask):
                dom_ids[k] = np.nan
                continue
            lab_vals = a[mask].astype(np.int64)
            lab_mode = np.bincount(lab_vals).argmax()
            dom_ids[k] = lab_mode
        dom_admin = dom_ids

    # build table
    rows: List[Dict] = []
    for k in ids:
        row = {
            "cluster_id": int(k),
            "area_km2": float(area_km2[k]),
            "mean_travel_min": float(tt_mean[k]) if np.isfinite(tt_mean[k]) else np.nan,
        }
        if pop_sum is not None: row["pop"] = float(pop_sum[k])
        if ag_km2 is not None:  row["cropland_km2"] = float(ag_km2[k])
        if pct_elec is not None: row["pct_electrified"] = float(pct_elec[k])
        if pct_rur is not None:  row["pct_rural"] = float(pct_rur[k])
        if dr_mean is not None:  row["drought_mean_0_1"] = float(dr_mean[k])
        if risk_cnt is not None: row["risk_roadcells"] = int(risk_cnt[k])
        if flood_mean is not None: row["flood_depth_mean_m"] = float(flood_mean[k])
        if rwi_mean is not None: row["rwi_mean"] = float(rwi_mean[k])

        # Shares of cluster area within travel-time thresholds (default from PARAMS.ISO_THRESH)
        iso_thresholds = tuple(getattr(PARAMS, "ISO_THRESH", (30, 60, 120)))
        mask_k = (lbl == k)
        total_cells = int(np.sum(mask_k))
        if total_cells > 0:
            tt_vals = tt.values  # minutes
            for thr in iso_thresholds:
                share = float(np.sum(mask_k & (tt_vals <= float(thr))) / total_cells)
                row[f"share_le_{int(thr)}m"] = share

        # centroid (lon,lat) from grid centers
        yy, xx = np.where(lbl == k)
        if yy.size > 0:
            lon = float(np.mean(T.x.values[xx]))
            lat = float(np.mean(T.y.values[yy]))
            row["centroid_lon"] = lon
            row["centroid_lat"] = lat

        # dominant municipality
        if dom_admin is not None and np.isfinite(dom_admin[k]):
            adm_label = int(dom_admin[k])
            meta = admin2_lut.loc[admin2_lut["ADM2_label"] == adm_label]
            if not meta.empty:
                row["ADM2CD_c"] = str(meta["ADM2CD_c"].iloc[0])
                row["NAM_1"]    = str(meta["NAM_1"].iloc[0])
                row["NAM_2"]    = str(meta["NAM_2"].iloc[0])

        rows.append(row)

    df = pd.DataFrame(rows).sort_values(by=["area_km2","pop"], ascending=[False, False], na_position="last")
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------- main -------------------------------------

def main() -> None:
    """
    Build connected-component clusters from the priority mask, prune tiny clusters,
    compute cluster KPIs, and save a label raster + cluster table.
    """
    # Template grid
    T = rxr.open_rasterio(PARAMS.TARGET_GRID, masked=True).squeeze()
    tf = T.rio.transform(); resx, resy = abs(tf.a), abs(tf.e)
    log.info(f"Template grid | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}")

    # Priority mask (required)
    mask = _open_align(Path(PRIORITY_TOP10_TIF), T, "nearest")
    if mask is None:
        raise FileNotFoundError(f"Priority mask not found: {PRIORITY_TOP10_TIF}")

    # Required rasters for KPIs
    pop   = _open_align(PATHS.OUT_R / f"{AOI}_pop_1km.tif", T, "bilinear")
    cropf = _open_align(PATHS.OUT_R / f"{AOI}_cropland_fraction_1km.tif", T, "bilinear")
    grid  = _open_align(PATHS.OUT_R / f"{AOI}_elec_grid_1km.tif", T, "nearest")
    rural = _open_align(PATHS.OUT_R / f"{AOI}_rural_1km.tif", T, "nearest")

    missing = [k for k,da in {"pop":pop,"cropf":cropf,"grid":grid,"rural":rural}.items() if da is None]
    if missing:
        raise RuntimeError(f"Missing required rasters for stats: {missing}")

    # Optional context rasters
    drt   = _open_align(PATHS.OUT_R / f"{AOI}_drought_1km.tif", T, "bilinear")
    risk  = _open_align(PATHS.OUT_R / f"{AOI}_roads_flood_risk_cells_1km.tif", T, "nearest")
    flood = _open_align(PATHS.OUT_R / f"{AOI}_flood_rp100_maxdepth_1km.tif", T, "bilinear")
    # Relative Wealth Index (optional; aligned in Step 00 if present)
    rwi = _open_align(out_r("rwi_meta_1km"), T, "bilinear")

    # Admin2 labels (for dominant municipality)
    admin2_lbl, admin2_lut = _admin2_label_grid(T)

    # 1) Connected components on the mask
    lbl_da = _connected_components(mask)

    # 2) Prune tiny clusters (by cells and by km²)
    lbl_da = _prune_clusters(lbl_da, T, min_cells=MIN_CLUSTER_CELLS, min_km2=MIN_CLUSTER_KM2)

    # 3) Renumber surviving clusters to 1..K (compact IDs)
    lbl = lbl_da.values
    uniq = np.unique(lbl); uniq = uniq[uniq > 0]
    remap = {old: i+1 for i, old in enumerate(uniq)}
    lbl_re = np.zeros_like(lbl, dtype=np.int32)
    for old, new in remap.items():
        lbl_re[lbl == old] = new
    lbl_da = xr.DataArray(lbl_re, coords=lbl_da.coords, dims=lbl_da.dims)

    # 4) Compute stats
    ras = {
        "pop": pop, "cropf": cropf, "grid": grid, "rural": rural,
        "drt": drt, "risk": risk, "flood": flood, "rwi": rwi
    }
    df = _cluster_stats(lbl_da, T, ras, admin2_lbl, admin2_lut)

    # 5) Write outputs
    if WRITE_CLUSTER_RASTER:
        write_gtiff(lbl_da, PRIORITY_CLUSTERS_TIF, like=T, nodata=0)
        # legacy name for compatibility if any down-stream expects it
        if WRITE_PER_CELL_CLUSTER_ID:
            write_gtiff(lbl_da, PRIORITY_CLUSTERS_TIF, like=T, nodata=0)

    out_csv = out_t("priority_clusters")
    df.to_csv(out_csv, index=False)
    log.info(f"Saved clusters → {Path(out_csv).name} | clusters={len(df)}")

    log.info("Step 11 complete.")


if __name__ == "__main__":
    main()

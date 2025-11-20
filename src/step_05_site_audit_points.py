"""
Step 05 — Site audit for the project points

Reads (AOI-prefixed / earlier steps)
------------------------------------
- PATHS.TRAVEL                      (target grid; minutes)
- {AOI}_veg_1km.tif                 (vegetation index)
- {AOI}_ntl_1km.tif                 (VIIRS night lights, 0..1)
- {AOI}_drought_1km.tif             (% severe drought)
- {AOI}_cropland_presence_1km.tif   (1 if any cropland in 1-km cell)
- {AOI}_cropland_fraction_1km.tif   (0..1 cropland cover per 1-km cell)
- {AOI}_elec_grid_1km.tif           (1 if grid-present cell)
- ROADS1K_TIF                       (main roads presence, from Step 04)
- PATHS.SITES                       (existing project points)

Writes
------
- SITE_AUDIT_CSV = outputs/tables/{AOI}_site_audit_points.csv

Notes
-----
- 5 km neighborhood = circular kernel radius of 5 cells (1 cell ≈ 1 km).
- cropland_km2_5km is area-true (fractional cropland × cell_area).
- electrified_share_5km is a cell-based share (# electrified cells / kernel cells).
- Added standard coordinate columns required by validator:
  * lon/lat  = site coordinates in EPSG:4326
  * x/y      = site coordinates in raster CRS (same as PATHS.TRAVEL)
  * row/col  = raster indices sampled
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.ndimage import convolve

from config import (
    PATHS, out_r, 
    SITE_AUDIT_CSV, 
    SITE_AUDIT_RADIUS_CELLS, 
    SITE_ID_FIELD, 
    ROADS1K_TIF, get_logger
)
from utils_geo import open_template, estimate_cell_area_km2
from rasterio.transform import rowcol

log = get_logger(__name__)


def _disk_kernel(radius_cells: int) -> np.ndarray:
    """Create a circular kernel (binary) with given radius in cells."""
    y, x = np.ogrid[-radius_cells:radius_cells + 1, -radius_cells:radius_cells + 1]
    return ((x**2 + y**2) <= radius_cells**2).astype(int)


def _assert_same_shape(*das) -> None:
    """Raise AssertionError if rasters do not share identical shape."""
    shapes = {da.shape for da in das}
    assert len(shapes) == 1, f"Rasters must share identical shape; got {shapes}"


def main() -> None:
    # ---------------------------------------------------------------------
    # Load base grid and all 1-km rasters
    # ---------------------------------------------------------------------
    T       = open_template(PATHS.TRAVEL)
    veg     = open_template(out_r("veg_1km"))
    ntl     = open_template(out_r("ntl_1km"))
    drt     = open_template(out_r("drought_1km"))
    clp     = open_template(out_r("cropland_presence_1km"))
    cl_frac = open_template(out_r("cropland_fraction_1km"))
    elg     = open_template(out_r("elec_grid_1km"))
    roads   = open_template(ROADS1K_TIF)

    _assert_same_shape(T, veg, ntl, drt, clp, cl_frac, elg, roads)

    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"Site audit inputs loaded | CRS={T.rio.crs} | "
        f"size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}"
    )

    # ---------------------------------------------------------------------
    # Load sites and prep neighborhood stats (5 km radius ≈ 5 cells)
    # ---------------------------------------------------------------------
    try:
        sites_xy  = gpd.read_file(PATHS.SITES).to_crs(T.rio.crs)     # raster CRS
        sites_ll  = gpd.read_file(PATHS.SITES).to_crs(4326)          # WGS84 lon/lat
    except Exception as e:
        raise RuntimeError(f"Failed to read sites from {PATHS.SITES}") from e

    if len(sites_xy) != len(sites_ll):
        raise RuntimeError("Sites length mismatch after CRS transforms (unexpected).")

    n_sites = len(sites_xy)
    log.info(f"Loaded {n_sites} site(s) from {PATHS.SITES.name}")

    radius = SITE_AUDIT_RADIUS_CELLS
    ker = _disk_kernel(radius)
    ker_cells = int(ker.sum())
    log.info(f"Neighborhood kernel: radius={radius} cell(s) | cells_in_kernel={ker_cells}")

    # Neighborhood sums (presence and fractional cropland)
    cl_nei      = convolve(np.nan_to_num(clp.values, nan=0),     ker, mode="constant", cval=0)
    elg_nei     = convolve(np.nan_to_num(elg.values, nan=0),     ker, mode="constant", cval=0)
    cell_km2    = estimate_cell_area_km2(cl_frac)
    cl_nei_km2  = convolve(np.nan_to_num(cl_frac.values, nan=0), ker, mode="constant", cval=0) * cell_km2

    # ---------------------------------------------------------------------
    # Sample per site
    # ---------------------------------------------------------------------
    rows: list[Dict[str, Any]] = []
    skipped = 0

    for idx, rec in sites_xy.iterrows():
        pt_xy = rec.geometry
        if pt_xy.is_empty or pt_xy.geom_type != "Point":
            skipped += 1
            continue

        # WGS84 counterpart (same index)
        try:
            pt_ll = sites_ll.at[idx, "geometry"]
        except Exception:
            pt_ll = None

        # Raster indices
        r, c = rowcol(T.rio.transform(), pt_xy.x, pt_xy.y)

        # Guard: point outside the grid
        if r < 0 or c < 0 or r >= T.rio.height or c >= T.rio.width:
            skipped += 1
            continue

        # Sample all layers via row/col (fast)
        tt = float(T.values[r, c])      # travel (minutes)
        vv = float(veg.values[r, c])    # vegetation index
        nn = float(ntl.values[r, c])    # night lights
        dd = float(drt.values[r, c])    # drought %

        # --- Site identifier: prefer configured field (e.g., "no"); fallback to row index ---
        sid = None
        if SITE_ID_FIELD:
            try:
                sid = rec.get(SITE_ID_FIELD, None)
            except Exception:
                sid = rec[SITE_ID_FIELD] if (SITE_ID_FIELD in rec) else None
        if sid is None or (isinstance(sid, str) and not sid.strip()):
            site_id_value = idx
        else:
            try:
                site_id_value = int(sid)
            except Exception:
                site_id_value = str(sid)

        # Build output row with standard coordinate columns
        row: Dict[str, Any] = {
            "site_id": site_id_value,
            # standard XY in raster CRS
            "x": float(pt_xy.x),
            "y": float(pt_xy.y),
            # standard lon/lat in EPSG:4326 (if available)
            "lon": float(pt_ll.x) if pt_ll is not None else np.nan,
            "lat": float(pt_ll.y) if pt_ll is not None else np.nan,
            # sampled cell address
            "row": int(r),
            "col": int(c),

            "travel_min": tt,
            "veg_index": vv,
            "ntl": nn,
            "drought_pct": dd,
            # Cropland around site (both cell-count and area-true)
            "cropland_cells_5km": int(cl_nei[r, c]),
            "cropland_km2_5km": float(cl_nei_km2[r, c]),
            # Electrification around site (cell-count + share within the 5 km kernel)
            "electrified_cells_5km": int(elg_nei[r, c]),
            "electrified_share_5km": (float(elg_nei[r, c]) / ker_cells) if ker_cells > 0 else np.nan,
            # Quick proximity flag to main road (1 km cell)
            "near_road_flag": int(np.nan_to_num(roads.values[r, c], nan=0)),
        }

        rows.append(row)

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    out_df = pd.DataFrame(rows)
    if out_df.empty:
        log.warning(
            "Site audit produced NO rows. Likely causes:\n"
            "  • All sites fall outside the target grid extent; or\n"
            "  • Site geometries are not Points; or\n"
            "  • The site layer CRS doesn’t match the rasters.\n"
            "Quick peek (first 2 sites in raster CRS):"
        )
        try:
            preview = sites_xy.head(2)[["geometry"]]
            for i, rec in preview.iterrows():
                g = rec.geometry
                if g and g.geom_type == "Point":
                    log.warning(f"  - site[{i}]: x={g.x:.5f}, y={g.y:.5f}")
                else:
                    log.warning(f"  - site[{i}]: non-Point or empty geometry")
        except Exception as e:
            log.warning(f"  (CRS preview failed: {e})")
    
    out_df.to_csv(SITE_AUDIT_CSV, index=False)
    log.info(
        f"Saved site audit → {SITE_AUDIT_CSV} | "
        f"rows={len(out_df)} | skipped_outside_grid={skipped}"
    )

    log.info("Step 05 complete.")


if __name__ == "__main__":
    main()

"""
Step 03 — Composite priority surface [0..1] and top-decile mask.

Reads (AOI-prefixed, produced by Step 00):
  - {AOI}_pop_1km.tif                  (persons per 1-km cell)
  - {AOI}_ntl_1km.tif                  (0..1 night lights)
  - {AOI}_veg_1km.tif                  (0.001..1 vegetation index)
  - {AOI}_drought_1km.tif              (% severe drought, 0..100)
  - PATHS.TRAVEL                       (minutes; target grid)

Writes:
  - PRIORITY_TIF           = outputs/rasters/{AOI}_priority_score_0_1.tif
  - PRIORITY_TOP10_TIF     = outputs/rasters/{AOI}_priority_top10_mask.tif

Formula:
  priority = w_acc*Access + w_pop*Population + w_veg*Vegetation
             + w_ntl*NightLights + w_drt*(1 - Drought%)

Notes:
  - Access = higher when minutes are lower (0 min → 1, 240+ min → 0).
  - Population is capped at P95 before normalization to reduce outlier dominance.
  - Night lights are capped at 0.3 (rural context) before normalization.
  - Drought % is clamped to [0,30] and inverted (lower drought → higher score).
  - NaNs are preserved from inputs into outputs.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import xarray as xr
from rasterio.enums import Resampling

from config import (
    PRIORITY_TIF_V1, 
    PRIORITY_TOP10_TIF_V1, 
    PATHS, PARAMS, out_r, 
    get_logger,
)
from utils_geo import (
    open_template, normalize_linear, percentile_cap, 
    write_gtiff_masked, apply_aoi_mask_if_enabled,
    select_top_mask_nan as select_top,
)

log = get_logger(__name__)


def _geo_like(da: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
    da2 = da.assign_coords({ref.rio.y_dim: ref[ref.rio.y_dim], ref.rio.x_dim: ref[ref.rio.x_dim]}).transpose(*ref.dims)
    try:
        da2.rio.write_crs(ref.rio.crs, inplace=True)
        da2.rio.write_transform(ref.rio.transform(), inplace=True)
    except Exception:
        pass
    return da2


def _clamp_travel(da: xr.DataArray, maxv: float = 240) -> xr.DataArray:
    """Clamp travel time to [0, maxv] (preserves NaN)."""
    return xr.where(da.isnull(), np.nan, xr.where(da > maxv, maxv, xr.where(da < 0, 0, da)))


def _assert_same_shape(*das: xr.DataArray) -> None:
    """Raise AssertionError if rasters do not share identical shape."""
    shapes = {da.shape for da in das}
    assert len(shapes) == 1, f"Rasters must share identical shape; got {shapes}"


def _coerce_like(da: xr.DataArray, ref: xr.DataArray, *, resamp: Resampling = Resampling.bilinear) -> xr.DataArray:
    """Put `da` on exactly the same grid as `ref` (CRS, transform, coords, dims)."""
    # If da has no CRS but already same shape: borrow from ref
    if getattr(da.rio, "crs", None) is None:
        if da.shape == ref.shape:
            da = da.rio.write_crs(ref.rio.crs, inplace=False)
            da = da.rio.write_transform(ref.rio.transform(), inplace=False)
        else:
            # Needs a real reprojection, but we can’t without a source CRS
            raise ValueError("Input raster missing CRS and shape differs from target; cannot coerce.")

    # If grid differs, reproject to match target grid exactly
    try:
        if (da.rio.crs != ref.rio.crs) or (da.rio.transform() != ref.rio.transform()) or (da.shape != ref.shape):
            da = da.rio.reproject_match(ref, resampling=resamp)
    except Exception:
        # Fall back to reproject_match if transform comparison is finicky
        da = da.rio.reproject_match(ref, resampling=resamp)

    # Force identical coords/dims to prevent future alignment
    da = da.assign_coords({ref.rio.y_dim: ref[ref.rio.y_dim], ref.rio.x_dim: ref[ref.rio.x_dim]}).transpose(*ref.dims)
    try:
        da.rio.write_crs(ref.rio.crs, inplace=True)
        da.rio.write_transform(ref.rio.transform(), inplace=True)
    except Exception:
        pass
    return da


def main() -> None:
    # -------------------------------------------------------------------------
    # Load rasters on the target grid
    # -------------------------------------------------------------------------
    T   = open_template(PATHS.TRAVEL)

    pop = _coerce_like(open_template(out_r("pop_1km")),     T, resamp=Resampling.bilinear)
    ntl = _coerce_like(open_template(out_r("ntl_1km")),     T, resamp=Resampling.bilinear)
    veg = _coerce_like(open_template(out_r("veg_1km")),     T, resamp=Resampling.bilinear)
    drt = _coerce_like(open_template(out_r("drought_1km")), T, resamp=Resampling.bilinear)

    _assert_same_shape(T, pop, ntl, veg, drt)

    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"Priority surface inputs loaded | CRS={T.rio.crs} | "
        f"size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}"
    )
    log.info(
        f"Weights | ACC={PARAMS.W_ACC:.2f} POP={PARAMS.W_POP:.2f} "
        f"VEG={PARAMS.W_VEG:.2f} NTL={PARAMS.W_NTL:.2f} DRT={PARAMS.W_DRT:.2f}"
    )

    # -------------------------------------------------------------------------
    # Build standardized components
    # -------------------------------------------------------------------------
    # Access: lower minutes → higher score (0..1)
    acc = 1 - normalize_linear(_clamp_travel(T, 240), 0, 240)

    # Population: cap at P95, then normalize to [0..1]
    pop_capped, p95 = percentile_cap(pop, 95)
    pop_s = normalize_linear(pop_capped, 0, p95 if p95 > 0 else 1)

    # Vegetation: 0.3..1.0 → [0..1] (tune min as needed)
    veg_s = normalize_linear(veg, 0.3, 1.0)

    # Night lights: cap at 0.3 (rural), 0..0.3 → [0..1]
    ntl_capped = xr.where(ntl.isnull(), np.nan, xr.where(ntl > 0.3, 0.3, ntl))
    ntl_s = normalize_linear(ntl_capped, 0, 0.3)

    # Drought: clamp 0..30% and invert → higher is better (less drought)
    drt_clamped = xr.where(drt.isnull(), np.nan, xr.where(drt < 0, 0, xr.where(drt > 30, 30, drt)))
    drt_pen = 1 - (drt_clamped / 30.0)

    # -------------------------------------------------------------------------
    # Composite priority and top decile
    # -------------------------------------------------------------------------
    priority = (
        PARAMS.W_ACC * acc +
        PARAMS.W_POP * pop_s +
        PARAMS.W_VEG * veg_s +
        PARAMS.W_NTL * ntl_s +
        PARAMS.W_DRT * drt_pen
    )
    # Mask at sink (policy-aware) to ensure NaN outside AOI
    priority = _geo_like(priority, T)
    priority = apply_aoi_mask_if_enabled(priority, T)  # central policy
    write_gtiff_masked(priority, PRIORITY_TIF_V1, like=T, nodata=np.nan)
    log.info(f"Wrote {PRIORITY_TIF_V1.name}")

    # Top 10% selection with NaN outside AOI
    top10 = select_top(priority, T, top_pct=0.10)      # float32 with NaN outside
    write_gtiff_masked(top10, PRIORITY_TOP10_TIF_V1, like=T, nodata=np.nan)

    # Quick summary: share of valid cells flagged as top decile
    total = int(np.isfinite(priority.values).sum())

    # top10 is a float mask with NaN outside AOI; treat values == 1 as selected
    flagged = int(np.nansum((top10.values == 1).astype(np.int64)))

    thr = float(np.nanpercentile(priority.values, 90))
    pct = 100.0 * flagged / total if total else np.nan

    log.info(f"Wrote {PRIORITY_TOP10_TIF_V1.name} | P90={thr:.3f} | top10 cells: {flagged}/{total} ({pct:.1f}%)")

    log.info("Step 03 complete.")


if __name__ == "__main__":
    main()

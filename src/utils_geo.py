"""
Geospatial helpers for raster-aligned processing (rasterio/rioxarray/xarray).

All functions work on the target grid defined in config.PARAMS.TARGET_GRID.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import rasterio as rio
import rioxarray as rxr
import xarray as xr
import geopandas as gpd
from rasterio.features import rasterize
from affine import Affine
from rasterio.enums import Resampling as _Resampling
from pathlib import Path  
from functools import lru_cache

from config import get_logger
log = get_logger(__name__)


def open_template(path: str | rio.PathLike) -> xr.DataArray:
    """
    Open a raster into a single-band xarray.DataArray with .rio accessor.
    Masked values become NaN.
    """
    da = rxr.open_rasterio(path, masked=True).squeeze()
    return da


def _to_resampling(resampling: str | _Resampling) -> _Resampling:
    """
    Convert a string like 'nearest'/'bilinear'/'max' to rasterio.enums.Resampling.
    If an enum is provided, pass it through unchanged.
    """
    if isinstance(resampling, _Resampling):
        return resampling
    if isinstance(resampling, str):
        try:
            return getattr(_Resampling, resampling)
        except AttributeError as e:
            raise ValueError(
                f"Unknown resampling '{resampling}'. "
                f"Valid: {[r.name for r in _Resampling]}"
            ) from e
    raise TypeError("resampling must be a str or rasterio.enums.Resampling")


def align_to_template(
    src_path: str | rio.PathLike,
    template_da: xr.DataArray,
    resampling: str | _Resampling = "nearest"
) -> xr.DataArray:
    """
    Reproject and align `src_path` raster to match `template_da` grid.
    Accepts resampling as string ('nearest', 'bilinear', 'max', ...) or Resampling enum.
    """
    src = rxr.open_rasterio(src_path, masked=True).squeeze()
    out = src.rio.reproject_match(template_da, resampling=_to_resampling(resampling))
    return out


def rasterize_vector(
        vec_path: str | rio.PathLike,
        template_da: xr.DataArray,
        field: Optional[str] = None,
        burn_value: float = 1.0,
        where: Optional[str] = None,
        dtype: str = "float32",
        all_touched: bool = True
    ) -> xr.DataArray:
    """
    Rasterize a vector layer to the template grid.
    If `field` is provided and exists, its values are burned; else constant `burn_value`.
    A pandas-like `where` query can filter features before rasterization.
    Set `all_touched=True` to better capture thin lines (roads, rivers).
    """
    gdf = gpd.read_file(vec_path)
    if where:
        gdf = gdf.query(where)

    # Ensure CRS matches template
    if gdf.crs is None:
        gdf.set_crs(template_da.rio.crs, inplace=True)
    else:
        gdf = gdf.to_crs(template_da.rio.crs)

    transform = template_da.rio.transform()
    out_shape: tuple[int, int] = (template_da.rio.height, template_da.rio.width)

    # Empty vector → full-NaN raster with correct georeferencing
    if gdf.empty:
        arr = np.full(out_shape, np.nan, dtype=dtype)
    else:
        if field and field in gdf.columns:
            shapes = [(geom, float(val)) for geom, val in zip(gdf.geometry, gdf[field])]
        else:
            shapes = [(geom, float(burn_value)) for geom in gdf.geometry]
        arr = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=np.nan,
            dtype=dtype,
            all_touched=all_touched,
        )

    da = xr.DataArray(arr, coords={"y": template_da.y, "x": template_da.x}, dims=("y","x"))
    da.rio.write_crs(template_da.rio.crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da


def rasterize_admin2_ids(
    gdf: gpd.GeoDataFrame,
    template: xr.DataArray,
    code_field: str = "ADM2CD_c",
    name1_field: str = "NAM_1",
    name2_field: str = "NAM_2",
    all_touched: bool = False,
):
    """
    Burn integer labels (1..N) per Admin2 polygon onto the template grid.
    Returns (idgrid_da, lookup_df). Background is 0.
    """
    # Align CRS
    try:
        if gdf.crs is not None and template.rio.crs is not None and gdf.crs != template.rio.crs:
            gdf = gdf.to_crs(template.rio.crs)
    except Exception:
        pass

    # Ensure key fields (fallback if missing)
    if code_field not in gdf.columns:
        gdf[code_field] = gdf.get("NAM_2", None)

    keep = [c for c in (code_field, name1_field, name2_field, "geometry") if c in gdf.columns]
    gdf = gdf[keep].dropna(subset=[code_field]).drop_duplicates(code_field).sort_values(code_field).copy()

    labels = np.arange(1, len(gdf) + 1, dtype="int32")
    gdf["_lab"] = labels

    shapes = list(zip(gdf.geometry, gdf["_lab"].astype(int)))
    out = rasterize(
        shapes=shapes,
        out_shape=template.shape,
        transform=template.rio.transform(),
        fill=0,
        all_touched=all_touched,
        dtype="int32",
    )

    idgrid = xr.DataArray(out, coords=template.coords, dims=template.dims)
    idgrid.rio.write_crs(template.rio.crs, inplace=True)
    idgrid.rio.write_transform(template.rio.transform(), inplace=True)

    lut = gdf[["_lab"] + [c for c in (code_field, name1_field, name2_field) if c in gdf.columns]].rename(
        columns={"_lab": "lab", code_field: "ADM2CD_c", name1_field: "NAM_1", name2_field: "NAM_2"}
    )
    return idgrid, lut


def _rasterize_aoi_mask(template_da: xr.DataArray) -> xr.DataArray | None:
    """
    Rasterize AOI polygon to the template grid. Returns a 0/1 mask (1 inside AOI, 0 outside).
    Returns None if AOI_BND missing.
    """
    try:
        from config import AOI_BND  # module-level path to AOI boundary
        aoi_vec = AOI_BND
    except Exception:
        aoi_vec = None

    if not aoi_vec or not Path(aoi_vec).exists():
        return None

    # Read & align CRS
    gdf = gpd.read_file(aoi_vec)
    if gdf.crs is None:
        gdf = gdf.set_crs(template_da.rio.crs)
    else:
        gdf = gdf.to_crs(template_da.rio.crs)

    transform = template_da.rio.transform()
    out_shape = template_da.shape

    # 0 outside, 1 inside (int8 to be explicit)
    arr = rasterize(
        shapes=[(geom, 1) for geom in gdf.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,     # safer for coastlines/slivers
    )

    da = xr.DataArray(arr, coords=template_da.coords, dims=template_da.dims)
    da.rio.write_crs(template_da.rio.crs, inplace=True)
    da.rio.write_transform(transform, inplace=True)
    return da


@lru_cache(maxsize=4)
def _cached_aoi_mask(shape: tuple, transform: tuple, crs: str | None) -> xr.DataArray | None:
    """
    Cache AOI mask by grid signature to avoid repeated rasterize cost.
    """
    # Build a minimal template from signature
    y, x = shape
    T = xr.DataArray(np.zeros((y, x), dtype=np.float32), dims=("y","x"))
    # Rebuild transform/CRS
    try:
        import affine
        A = affine.Affine(*transform)
    except Exception:
        A = None
    if A is not None:
        T.rio.write_transform(A, inplace=True)
    if crs:
        T.rio.write_crs(crs, inplace=True)
    return _rasterize_aoi_mask(T)

def _same_grid(a: xr.DataArray, b: xr.DataArray) -> bool:
    try:
        return (
            a.shape == b.shape and
            a.rio.crs == b.rio.crs and
            a.rio.transform() == b.rio.transform()
        )
    except Exception:
        return False

def _is_float_da(da: xr.DataArray) -> bool:
    return np.issubdtype(da.dtype, np.floating)

def _mask_policy() -> str:
    """
    Resolve the effective mask policy. Falls back to legacy toggles if needed.
    """
    try:
        from config import MASK_POLICY
        pol = str(MASK_POLICY).lower().strip()
        if pol in {"defer", "eager", "none"}:
            return pol
    except Exception:
        pass
    # Legacy fallback: emulate prior behavior with MASK_OUTSIDE_AOI
    try:
        from config import MASK_OUTSIDE_AOI
        return "eager" if bool(MASK_OUTSIDE_AOI) else "none"
    except Exception:
        return "none"


def _mask_overlap_pct(mask: xr.DataArray) -> float:
    """
    Share of target grid cells that are inside AOI (mask == 1) over ALL grid cells.
    Assumes AOI mask is 0/1. If mask contains NaNs everywhere, return 0.
    """
    m = np.asarray(mask.values)
    if m.size == 0:
        return 0.0
    if np.isnan(m).all():
        return 0.0
    inside = np.sum(m == 1)
    total  = m.size
    return float(inside) / float(total) if total > 0 else 0.0


def _should_skip_mask_for_overlap(mask: xr.DataArray) -> tuple[bool, float]:
    """
    Check overlap against configured threshold.
    """
    try:
        from config import AOI_BND_VALIDATE, AOI_MIN_OVERLAP_PCT
        if not AOI_BND_VALIDATE:
            return (False, 1.0)
        pct = _mask_overlap_pct(mask)
        return (pct < float(AOI_MIN_OVERLAP_PCT), pct)
    except Exception:
        # If config missing, err on the side of applying the mask
        return (False, 1.0)


def apply_aoi_mask_if_enabled(da: xr.DataArray, template_like: xr.DataArray, *, force_mask: Optional[bool] = None) -> xr.DataArray:
    """
    Policy-aware AOI mask application that:
      - ALWAYS aligns `da` and AOI mask to `template_like` first,
      - Skips masking if policy says so, or if AOI has ~no overlap,
      - Applies mask via raw NumPy to avoid alignment pitfalls.

    Parameters
    ----------
    da : xr.DataArray
        Raster to (optionally) mask.
    template_like : xr.DataArray
        Target grid reference (CRS/transform/coords/dims).
    force_mask : Optional[bool]
        - True  -> force masking regardless of MASK_POLICY
        - False -> skip masking regardless of MASK_POLICY
        - None  -> follow MASK_POLICY
    """
    # 0) Resolve policy
    policy = _mask_policy()
    if force_mask is False:
        policy_effective = "none"
    elif force_mask is True:
        policy_effective = "eager"
    else:
        policy_effective = policy  # "defer" | "eager" | "none"

    if policy_effective == "none":
        return da

    # 1) Build/cached AOI mask on the template grid signature
    tf = template_like.rio.transform()
    crs = str(template_like.rio.crs) if template_like.rio.crs else None
    sig = (template_like.shape, tuple(tf) if tf is not None else (), crs)
    mask = _cached_aoi_mask(*sig)
    if mask is None:
        return da

    # 2) Ensure `da` has georef; if missing, borrow from template_like
    if getattr(da.rio, "crs", None) is None:
        da = da.rio.write_crs(template_like.rio.crs, inplace=False)
    try:
        _ = da.rio.transform(recalc=True)
    except Exception:
        da = da.rio.write_transform(template_like.rio.transform(), inplace=False)

    # 3) Put both on the template grid
    resamp = _Resampling.bilinear if _is_float_da(da) else _Resampling.nearest
    if not _same_grid(da, template_like):
        da = da.rio.reproject_match(template_like, resampling=resamp)
    if not _same_grid(mask, template_like):
        mask = mask.rio.reproject_match(template_like, resampling=_Resampling.nearest)

    ydim, xdim = template_like.rio.y_dim, template_like.rio.x_dim
    da   = da.assign_coords({ydim: template_like[ydim], xdim: template_like[xdim]}).transpose(*template_like.dims)
    mask = mask.assign_coords({ydim: template_like[ydim], xdim: template_like[xdim]}).transpose(*template_like.dims)

    # 4) Overlap guard
    skip, pct = _should_skip_mask_for_overlap(mask)
    if skip:
        log.warning(
            "AOI mask skipped: overlap too small (overlap_pct=%.4f). "
            "Check AOI_BND, CRS, or AOI token. Set MASK_POLICY='none' to silence.",
            pct
        )
        return da

    # 5) Apply mask (1 inside AOI → keep; else NaN)
    masked_vals = np.where(mask.values == 1, da.values, np.nan)
    out = xr.DataArray(masked_vals, coords=template_like.coords, dims=template_like.dims, attrs=da.attrs)
    try:
        out.rio.write_crs(template_like.rio.crs, inplace=True)
        out.rio.write_transform(template_like.rio.transform(), inplace=True)
    except Exception:
        pass
    return out


def build_aoi_mask(template_like: xr.DataArray) -> xr.DataArray | None:
    """
    Public wrapper to get AOI mask aligned to `template_like`.
    Returns a DataArray where 1=inside AOI and NaN outside (and off-grid).
    Respects cached construction for speed.
    """
    tf = template_like.rio.transform()
    crs = str(template_like.rio.crs) if template_like.rio.crs else None
    sig = (template_like.shape, tuple(tf) if tf is not None else (), crs)
    m = _cached_aoi_mask(*sig)
    if m is None:
        return None
    # Snap exactly to template grid and promote to {1, NaN}
    if not _same_grid(m, template_like):
        m = m.rio.reproject_match(template_like, resampling=_Resampling.nearest)
    m = m.assign_coords(template_like.coords).transpose(*template_like.dims)
    return m.where(m == 1, np.nan)


def _area_da_for(template_like: xr.DataArray) -> xr.DataArray:
    """
    Return a 2-D DataArray of cell areas in km^2 aligned to template_like.
    Uses spherical approx for EPSG:4326 (lat/lon), otherwise affine pixel area.
    """
    try:
        # attempt to detect EPSG:4326
        crs = template_like.rio.crs
        if crs and ("4326" in str(crs).lower()):
            return cell_area_km2_latlon(template_like)
    except Exception:
        pass
    # fallback: affine-based constant area for projected CRS
    a_km2 = estimate_cell_area_km2(template_like)
    A = np.full(template_like.shape, a_km2, dtype="float32")
    da = xr.DataArray(A, coords=template_like.coords, dims=template_like.dims)
    try:
        da.rio.write_crs(template_like.rio.crs, inplace=True)
        da.rio.write_transform(template_like.rio.transform(), inplace=True)
    except Exception:
        pass
    return da


def select_top_mask_nan(
    score: xr.DataArray,
    template_like: xr.DataArray,
    *,
    top_pct: float | None = None,
    top_km2: float | None = None
) -> xr.DataArray:
    """
    Build a NaN-preserving selection mask aligned to template_like.

    - Where score is NaN -> NaN in output.
    - Among finite cells, mark 1 for selected, 0 otherwise.
    - If top_km2 is provided, select by cumulative area (km^2).
      Else uses top_pct (0..1, default 0.10) over valid cells.

    Returns float32 DataArray so NaN is representable.
    """
    v = np.asarray(score.values)
    out = np.full_like(v, np.nan, dtype="float32")

    valid = np.isfinite(v)
    if not valid.any():
        return xr.DataArray(out, coords=score.coords, dims=score.dims)

    # Initialize valid cells to 0; fill 1 for selected later
    out[valid] = 0.0

    if top_km2 is not None:
        area = _area_da_for(template_like).values.astype("float64")
        area[~valid] = 0.0
        # sort valid cells by score descending
        flat_idx = np.argsort(v[valid])[::-1]
        a_sorted = area[valid][flat_idx]
        cum_a = np.cumsum(a_sorted)
        k = int(np.searchsorted(cum_a, float(top_km2), side="left")) + 1
        rr, cc = np.where(valid)
        sel_r, sel_c = rr[flat_idx[:k]], cc[flat_idx[:k]]
        out[sel_r, sel_c] = 1.0
    else:
        pct = 0.10 if (top_pct is None) else float(top_pct)
        thr = np.nanpercentile(v, (1.0 - pct) * 100.0)
        pick = valid & (v >= thr)
        out[pick] = 1.0

    return xr.DataArray(out, coords=score.coords, dims=score.dims)


def remove_small_clusters(mask_da: xr.DataArray, min_cells: int) -> xr.DataArray:
    """
    Remove 8-connected components smaller than `min_cells`.
    Preserves NaN outside; operates only on finite (0/1) cells.

    Returns float32 (NaN outside, 0/1 inside).
    """
    if (min_cells or 0) <= 1:
        return mask_da.astype("float32")

    arr = mask_da.values
    is_nan = ~np.isfinite(arr)
    bin_arr = np.where(is_nan, 0, (arr > 0).astype(np.uint8))

    try:
        from scipy.ndimage import label
        lbl, n = label(bin_arr)
        if n == 0:
            out = bin_arr.astype("float32")
            out[is_nan] = np.nan
            return xr.DataArray(out, coords=mask_da.coords, dims=mask_da.dims)
        sizes = np.bincount(lbl.ravel())
        kill = np.where(sizes < int(min_cells))[0]
        kill = kill[kill != 0]
        if kill.size:
            pruned = bin_arr.copy()
            for lab in kill:
                pruned[lbl == lab] = 0
        else:
            pruned = bin_arr
        out = pruned.astype("float32")
        out[is_nan] = np.nan
        return xr.DataArray(out, coords=mask_da.coords, dims=mask_da.dims)
    except Exception:
        log.warning("remove_small_clusters: scipy not available; skipping cluster pruning.")
        return mask_da.astype("float32")
    

def write_gtiff_masked(
    da: xr.DataArray,
    out_path: Path | str,
    like: xr.DataArray,
    nodata=np.nan,
    *,
    force_mask: Optional[bool] = None
) -> None:
    """
    Convenience writer: apply AOI mask (policy-aware) then write GeoTIFF.

    force_mask:
      - True  -> mask regardless of MASK_POLICY
      - False -> never mask
      - None  -> follow MASK_POLICY
    """
    da2 = apply_aoi_mask_if_enabled(da, like, force_mask=force_mask)
    write_gtiff(da2, out_path, like=like, nodata=nodata)


    
def write_gtiff(
    da: xr.DataArray,
    path: str | rio.PathLike,
    nodata: float = np.nan,
    compress: str = "LZW",
    dtype: str = "float32",
    like: Optional[xr.DataArray] = None,
) -> None:
    """
    Write a DataArray as GeoTIFF with nodata and compression.
    If CRS/transform are missing, inherit from `like=` if provided.
    Otherwise, raise a clear error (never write without CRS).
    Strips conflicting CF keys (_FillValue) from attrs/encoding to avoid xarray errors.
    """
    da = da.copy()

    # Remove CF metadata that can collide
    da.attrs.pop("_FillValue", None)
    da.encoding.pop("_FillValue", None)

    # Robustly detect missing CRS/transform
    crs_missing = (da.rio.crs is None)
    try:
        _ = da.rio.transform(recalc=True)  # will raise if missing
        tf_missing = False
    except Exception:
        tf_missing = True

    if crs_missing or tf_missing:
        if like is None or like.rio.crs is None:
            raise ValueError(
                "DataArray has no CRS/transform. Pass `like=` with a georeferenced raster "
                "or write CRS/transform explicitly before saving."
            )
        da = da.rio.write_crs(like.rio.crs, inplace=False)
        da = da.rio.write_transform(like.rio.transform(), inplace=False)

    da = da.astype(dtype)
    da = da.rio.write_nodata(nodata, inplace=True)
    da.rio.to_raster(path, compress=compress, dtype=dtype)


def reclass_le(da: xr.DataArray, threshold: float) -> xr.DataArray:
    """Binary mask: 1 where da <= threshold, 0 where > threshold, NaN preserved."""
    out = xr.where(da.isnull(), np.nan, xr.where(da <= threshold, 1, 0))
    # Preserve coords/attrs & reattach CRS/transform so rioxarray keeps georeferencing
    out = out.assign_coords(da.coords).assign_attrs(da.attrs)
    try:
        out.rio.write_crs(da.rio.crs, inplace=True)
        out.rio.write_transform(da.rio.transform(), inplace=True)
    except Exception:
        pass
    return out


def zonal_sum(mask_da: xr.DataArray, value_da: xr.DataArray) -> float:
    """
    Sum `value_da` where `mask_da == 1` (NaN-safe). Returns a Python float.
    Raises AssertionError if shapes don't match.
    """
    assert mask_da.shape == value_da.shape, "mask/value rasters must have identical shape"
    m = (mask_da == 1)
    vals = value_da.where(m)
    return float(np.nansum(vals.values))


def zonal_mean_by_idgrid(value_da: xr.DataArray, idgrid_da: xr.DataArray):
    """
    Mean per integer label in `idgrid_da` (0 = background ignored).
    Returns dict: {label_int: mean_value}.
    """
    v = value_da.values
    ids = idgrid_da.values.astype("int32")
    mask = np.isfinite(v) & (ids > 0)
    if not mask.any():
        return {}

    vv = v[mask]
    ii = ids[mask]

    max_id = int(ii.max())
    sums = np.bincount(ii, weights=vv, minlength=max_id + 1)
    cnts = np.bincount(ii, minlength=max_id + 1)
    means = np.divide(sums, cnts, out=np.full_like(sums, np.nan, dtype="float64"), where=cnts > 0)

    return {lab: float(means[lab]) for lab in range(1, max_id + 1) if cnts[lab] > 0}


def percentile_cap(da: xr.DataArray, q: float = 95) -> tuple[xr.DataArray, float]:
    """
    Cap array at percentile q, returning (capped_da, cap_value).
    """
    vals = da.values
    if not np.isfinite(vals).any():
        capped = xr.zeros_like(da, dtype=float)
        # preserve geo
        capped = capped.assign_coords(da.coords).assign_attrs(da.attrs)
        try:
            capped.rio.write_crs(da.rio.crs, inplace=True)
            capped.rio.write_transform(da.rio.transform(), inplace=True)
        except Exception:
            pass
        return capped, 0.0

    p = np.nanpercentile(vals, q)
    capped = xr.where(da > p, p, da)

    # preserve geo
    capped = capped.assign_coords(da.coords).assign_attrs(da.attrs)
    try:
        capped.rio.write_crs(da.rio.crs, inplace=True)
        capped.rio.write_transform(da.rio.transform(), inplace=True)
    except Exception:
        pass
    return capped, float(p)


def normalize_linear(da: xr.DataArray, minv: float, maxv: float) -> xr.DataArray:
    """
    Linear rescale to [0,1] then clamp.
    If minv == maxv, returns a zero array.
    """
    if maxv == minv:
        out = xr.zeros_like(da, dtype=float)
    else:
        out = (da - minv) / (maxv - minv)
        out = xr.where(out < 0, 0, xr.where(out > 1, 1, out))

    # --- preserve coords/attrs + georeferencing
    out = out.assign_coords(da.coords).assign_attrs(da.attrs)
    try:
        out.rio.write_crs(da.rio.crs, inplace=True)
        out.rio.write_transform(da.rio.transform(), inplace=True)
    except Exception:
        pass
    return out


def _make_supersampled_grid(
        template_da: xr.DataArray, 
        supersample: int
    ) -> tuple[Affine, tuple[int,int]]:
    """
    Create an Affine transform and output shape for a supersampled grid
    that exactly covers the template extent (supersample^2 subcells per coarse pixel).
    """
    tf = template_da.rio.transform()
    height, width = template_da.rio.height, template_da.rio.width

    # Scale the affine for finer pixels
    a = tf.a / supersample         # pixel width
    e = tf.e / supersample         # pixel height (usually negative)
    supers_tf = Affine(a, tf.b, tf.c, tf.d, e, tf.f)
    supers_shape = (height * supersample, width * supersample)
    return supers_tf, supers_shape


def fractional_rasterize_polygon(
        vec_path: str | rio.PathLike,
        template_da: xr.DataArray,
        supersample: int = 10,
        where: Optional[str] = None
    ) -> xr.DataArray:
    """
    Approximate polygon *fraction per coarse cell* by supersampling.
    Steps:
      1) Rasterize polygons at (supersample ×) finer grid with burn=1.
      2) Block-average back to template grid → fraction in [0,1].
    Notes:
      - supersample=10 on a 1 km grid ~100 m subcells (good with 10 m source polygons).
      - If polygons have tiny slivers, consider increasing supersample to 20.
    """
    gdf = gpd.read_file(vec_path)
    if where:
        gdf = gdf.query(where)

    if gdf.crs is None:
        gdf.set_crs(template_da.rio.crs, inplace=True)
    else:
        gdf = gdf.to_crs(template_da.rio.crs)

    # Build fine grid
    fine_tf, (Hf, Wf) = _make_supersampled_grid(template_da, supersample)
    shapes = [(geom, 1.0) for geom in gdf.geometry]
    fine = rasterize(
        shapes=shapes, out_shape=(Hf, Wf), transform=fine_tf,
        fill=0.0, dtype="float32"
    )

    # Block reduce by mean → fraction per coarse cell
    ss = supersample
    H, W = template_da.rio.height, template_da.rio.width

    # Ensure float for averaging, then reshape into (H, ss, W, ss)
    fine = fine.astype(np.float32, copy=False).reshape(H, ss, W, ss)

    # Average over the supersample axes -> (H, W)
    frac = fine.mean(axis=(1, 3), dtype=np.float32)

    da = xr.DataArray(frac, coords={"y": template_da.y, "x": template_da.x}, dims=("y","x"))
    da.rio.write_crs(template_da.rio.crs, inplace=True)
    da.rio.write_transform(template_da.rio.transform(), inplace=True)
    return da


def estimate_cell_area_km2(template_da: xr.DataArray) -> float:
    """
    Estimate per-pixel area in km^2 using the affine transform.
    Assumes CRS units are meters. If units are degrees, returns 1.0 as a safe fallback.
    """
    tf = template_da.rio.transform()
    # Heuristic: if absolute pixel size is comfortably < 1e-2, assume degrees.
    resx, resy = abs(tf.a), abs(tf.e)
    if resx < 0.01 and resy < 0.01:
        # Geographic degrees — area varies; consider projecting beforehand.
        return 1.0
    return (resx * resy) / 1_000_000.0  # m^2 → km^2


def blocksum_to_template(src_path: str | rio.PathLike, template_da: xr.DataArray, factor: int) -> xr.DataArray:
    """
    Sum a finer grid into the coarser template grid by an integer factor (e.g., 100m -> 1km factor=10).
    Assumes same CRS & near-perfect alignment; for mismatched grids, reproject first, then coarsen.
    """
    fine = rxr.open_rasterio(src_path, masked=True).squeeze()
    # Reproject first to the *same* CRS as template to avoid degree/metric mismatch
    fine = fine.rio.reproject(template_da.rio.crs)
    # Snap to template extent (pad/trim so shapes are divisible by factor)
    H, W = fine.shape
    Ht, Wt = template_da.shape
    # Coarsen with exact factor (drop remainder)
    coarsened = fine.isel(
        y=slice(0, (H // factor) * factor),
        x=slice(0, (W // factor) * factor)
    ).coarsen(y=factor, x=factor, boundary="trim").sum()
    # Reproject_match to template for exact alignment (nearest is fine after sum)
    out = coarsened.rio.reproject_match(template_da, resampling=_Resampling.nearest)
    return out


def focal_mean(da: xr.DataArray, radius: int = 1) -> xr.DataArray:
    """
    Mean filter over a (2*radius+1) square kernel; preserves NaNs.

    Always returns a 2-D (y, x) DataArray, with CRS/transform preserved.
    """
    if radius <= 0:
        return da

    # Drop singleton dims, then ensure (y, x) order
    arr = da.squeeze(drop=True)
    if not {"y", "x"}.issubset(arr.dims):
        return da  # only defined for rasters with y/x
    arr = arr.transpose("y", "x")

    k = 2 * radius + 1
    sm = arr.rolling(y=k, x=k, center=True, min_periods=1).mean()

    # Carry georeferencing + attrs
    sm = sm.assign_coords(arr.coords).assign_attrs(arr.attrs)
    try:
        sm.rio.write_crs(arr.rio.crs, inplace=True)
        sm.rio.write_transform(arr.rio.transform(), inplace=True)
    except Exception:
        pass

    # Return as 2-D (y, x)
    return sm


def cell_area_km2_latlon(template_da: xr.DataArray) -> xr.DataArray:
    """
    Approximate cell area (km²) for a regular lat/lon grid in EPSG:4326.
    - Uses a simple spherical-earth approximation.
    - Returns a 2-D (y, x) DataArray aligned to the template.
    """
    y = template_da["y"].values
    x = template_da["x"].values
    ny, nx = y.size, x.size

    # Infer grid spacing in degrees (assumes evenly spaced coordinates)
    dlat = float(abs(y[1] - y[0])) if ny > 1 else 0.0
    dlon = float(abs(x[1] - x[0])) if nx > 1 else 0.0

    # Mean km per degree at the equator (spherical approx)
    KM_PER_DEG = 111.32

    # dx depends on latitude via cos(phi); dy is constant across the row
    phi = np.deg2rad(y)                    # length ny
    dx_km_row = KM_PER_DEG * dlon * np.cos(phi)   # (ny,)  km in x per pixel
    dy_km = KM_PER_DEG * dlat                     # scalar km in y per pixel

    # Broadcast to (ny, nx)
    # area = dx * dy for every column; replicate dx across columns
    area = np.repeat(dx_km_row[:, None], nx, axis=1) * dy_km  # (ny, nx)

    da = xr.DataArray(area, coords={"y": template_da.y, "x": template_da.x}, dims=("y", "x"))
    try:
        da.rio.write_crs(template_da.rio.crs, inplace=True)
        da.rio.write_transform(template_da.rio.transform(), inplace=True)
    except Exception:
        pass
    return da


def align_rwi_to_template(rwi_path: Path, template_da):
    """
    Align Meta RWI (typically ~2.4 km; value range ~[-2, +2]) to the 1-km template.
    Bilinear resampling is appropriate (continuous index).
    """
    return align_to_template(rwi_path, template_da, resampling="bilinear")


def normalize_rwi_to_equity(rwi_da, clip_pct=(1, 99)):
    """
    Convert RWI (where lower = poorer) to an equity score in [0,1] where 1 = poorer.
    Steps:
      1) percentile clip (default 1..99) to reduce outliers,
      2) min-max scale to [0,1],
      3) invert so higher means poorer (1 - scaled).

    Returns an xarray.DataArray on the same grid.
    """
    import numpy as np
    lo, hi = np.nanpercentile(rwi_da.values, [clip_pct[0], clip_pct[1]])
    r = rwi_da.clip(min=lo, max=hi)
    scaled = (r - lo) / (hi - lo + 1e-9)  # in [0,1]
    equity = 1.0 - scaled                 # 1 => poorer; 0 => wealthier
    equity.name = "rwi_equity01"
    return equity

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


def _rasterize_aoi_mask(template_da: xr.DataArray) -> xr.DataArray | None:
    """
    Rasterize AOI polygon to the template grid. Returns a 0/1 mask (1 inside AOI)
    with NaN outside grid. Returns None if AOI_BND missing.
    """
    try:
        from config import PATHS  # local import to avoid circulars at import time
        aoi_vec = getattr(PATHS, "AOI_BND", None)
    except Exception:
        aoi_vec = None

    if not aoi_vec or not Path(aoi_vec).exists():
        return None

    mask = rasterize_vector(aoi_vec, template_da, burn_value=1)  # 1 inside AOI; NaN elsewhere
    return mask

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

def apply_aoi_mask_if_enabled(da: xr.DataArray, template_like: xr.DataArray) -> xr.DataArray:
    """
    If PARAMS.MASK_OUTSIDE_AOI is True and AOI boundary exists,
    set values outside AOI to NaN (preserve true zeros inside AOI).
    """
    try:
        from config import PARAMS
    except Exception:
        return da

    if not getattr(PARAMS, "MASK_OUTSIDE_AOI", False):
        return da

    # Build grid signature for cache
    tf = template_like.rio.transform()
    crs = str(template_like.rio.crs) if template_like.rio.crs else None
    sig = (template_like.shape, tuple(tf) if tf is not None else (), crs)

    mask = _cached_aoi_mask(*sig)
    if mask is None:
        return da

    # Reproject mask to match da if needed (should already match if you pass the template)
    if (mask.shape != da.shape) or (mask.rio.transform() != da.rio.transform()) or (mask.rio.crs != da.rio.crs):
        mask = mask.rio.reproject_match(da, resampling="nearest")

    return da.where(mask == 1, np.nan)

def write_gtiff_masked(da: xr.DataArray, out_path: Path | str, like: xr.DataArray, nodata=np.nan) -> None:
    """
    Convenience writer: AOI-mask (if enabled), then write GeoTIFF.
    """
    da2 = apply_aoi_mask_if_enabled(da, like)
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

    # If da lacks CRS/transform, borrow from `like`
    if (getattr(da.rio, "crs", None) is None or
        getattr(da.rio, "transform", None) is None):
        if like is not None:
            if like.rio.crs is None:
                raise ValueError("`like` has no CRS; cannot inherit georeferencing.")
            da = da.rio.write_crs(like.rio.crs, inplace=False)
            da = da.rio.write_transform(like.rio.transform(), inplace=False)
        else:
            raise ValueError(
                "DataArray has no CRS/transform. "
                "Pass `like=` with a georeferenced raster or write CRS explicitly."
            )

    da = da.astype(dtype)
    da = da.rio.write_nodata(nodata, inplace=True)
    da.rio.to_raster(path, compress=compress, dtype=dtype)


def reclass_le(da: xr.DataArray, threshold: float) -> xr.DataArray:
    """Binary mask: 1 where da <= threshold, 0 where > threshold, NaN preserved."""
    return xr.where(da.isnull(), np.nan, xr.where(da <= threshold, 1, 0))


def zonal_sum(mask_da: xr.DataArray, value_da: xr.DataArray) -> float:
    """
    Sum `value_da` where `mask_da == 1` (NaN-safe). Returns a Python float.
    Raises AssertionError if shapes don't match.
    """
    assert mask_da.shape == value_da.shape, "mask/value rasters must have identical shape"
    m = (mask_da == 1)
    vals = value_da.where(m)
    return float(np.nansum(vals.values))


def percentile_cap(da: xr.DataArray, q: float = 95) -> tuple[xr.DataArray, float]:
    """
    Cap array at percentile q, returning (capped_da, cap_value).
    """
    p = np.nanpercentile(da.values, q)
    return xr.where(da > p, p, da), float(p)


def normalize_linear(da: xr.DataArray, minv: float, maxv: float) -> xr.DataArray:
    """
    Linear rescale to [0,1] then clamp.
    If minv == maxv, returns a zero array.
    """
    if maxv == minv:
        return xr.zeros_like(da, dtype=float)
    out = (da - minv) / (maxv - minv)
    return xr.where(out < 0, 0, xr.where(out > 1, 1, out))



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
    out = coarsened.rio.reproject_match(template_da, resampling="nearest")
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

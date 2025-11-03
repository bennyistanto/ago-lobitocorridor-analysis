"""
Step 00 — Align all base rasters to the travel-time grid and rasterize key vectors.

Outputs:
  - AOI-prefixed, 1 km-aligned rasters in outputs/rasters
  - Fractional cropland (0..1 per 1-km cell)
  - 1-km flood max-depth raster for quick screening
  - 1-km flood **exceedance fraction** (share of subpixels with depth ≥ threshold)
  - (Optional) Meta Relative Wealth Index aligned to 1-km if provided

Notes:
  - WorldPop is already ~1 km; we align with nearest.
  - Continuous rasters (NTL, vegetation, drought, RWI) use bilinear.
  - Flood is kept at native 30 m for engineering checks; we also make:
      (a) a 1-km "max depth" view, and
      (b) a 1-km **coverage fraction** of subpixels exceeding the risk threshold.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import geopandas as gpd
from rasterio.enums import Resampling

from config import PATHS, PARAMS, out_r, get_logger
from utils_geo import (
    open_template,
    align_to_template,
    rasterize_vector,
    write_gtiff_masked,
    fractional_rasterize_polygon,
    align_rwi_to_template,  # <- optional RWI helper
)

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger(__name__)


def _pick_field(vec_path: Path, candidates: list[str]) -> str | None:
    """
    Return the first attribute name that exists in the vector file.

    Examples:
      _pick_field(PATHS.ELEC, ["FinalElecCode2020", "FinalElecC"])
      _pick_field(PATHS.SETTLE, ["IsUrban"])
    """
    try:
        gdf = gpd.read_file(vec_path)
        cols = set(map(str, gdf.columns))
        for c in candidates:
            if c in cols:
                return c
        log.warning("None of the candidate fields %s found in %s (columns=%s)",
                    candidates, vec_path.name, sorted(cols))
        return None
    except Exception as e:
        log.warning("Failed reading %s to inspect fields (%s).", vec_path, e)
        return None


def main() -> None:
    """Align inputs to the travel-time grid and produce AOI-prefixed base rasters."""
    # -------------------------------------------------------------------------
    # Template grid
    # -------------------------------------------------------------------------
    T = open_template(PARAMS.TARGET_GRID)  # target grid (travel-time, 1 km)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"Loaded target grid | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | "
        f"cell={resx:.4f}x{resy:.4f} (units of CRS)"
    )

    # -------------------------------------------------------------------------
    # Raster alignment (1 km): POP (nearest), NTL/VEG/DROUGHT (bilinear)
    # -------------------------------------------------------------------------
    log.info("Aligning rasters to target grid...")
    pop_1km = align_to_template(PATHS.POP, T, resampling="nearest")
    ntl_1km = align_to_template(PATHS.NTL, T, resampling="bilinear")
    veg_1km = align_to_template(PATHS.VEG, T, resampling="bilinear")
    drt_1km = align_to_template(PATHS.DROUGHT, T, resampling="bilinear")

    write_gtiff_masked(pop_1km, out_r("pop_1km"), like=T, nodata=np.nan)
    log.info(f"Wrote {out_r('pop_1km').name}")
    write_gtiff_masked(ntl_1km, out_r("ntl_1km"), like=T, nodata=np.nan)
    log.info(f"Wrote {out_r('ntl_1km').name}")
    write_gtiff_masked(veg_1km, out_r("veg_1km"), like=T, nodata=np.nan)
    log.info(f"Wrote {out_r('veg_1km').name}")
    write_gtiff_masked(drt_1km, out_r("drought_1km"), like=T, nodata=np.nan)
    log.info(f"Wrote {out_r('drought_1km').name}")

    # -------------------------------------------------------------------------
    # OPTIONAL: Relative Wealth Index (Meta), align to 1-km grid
    # -------------------------------------------------------------------------
    try:
        if getattr(PATHS, "RWI", None) and Path(PATHS.RWI).exists():
            log.info("Aligning Meta Relative Wealth Index (RWI) to 1-km grid...")
            rwi_1km = align_rwi_to_template(PATHS.RWI, T)  # bilinear inside
            write_gtiff_masked(rwi_1km, out_r("rwi_meta_1km"), like=T, nodata=np.nan)
            log.info(f"Wrote {out_r('rwi_meta_1km').name}")
        else:
            log.info("No RWI file found; skipping equity layer alignment.")
    except Exception as e:
        log.warning(f"RWI alignment skipped due to error: {e}")

    # -------------------------------------------------------------------------
    # Cropland: presence and fractional coverage per 1-km cell
    # -------------------------------------------------------------------------
    log.info("Rasterizing cropland presence (1=any in cell)...")
    cl_pres = rasterize_vector(PATHS.CROPLAND, T, burn_value=1)
    write_gtiff_masked(cl_pres, out_r("cropland_presence_1km"), like=T, nodata=np.nan)
    log.info(f"Wrote {out_r('cropland_presence_1km').name}")

    log.info("Rasterizing cropland fraction (0..1 per 1-km cell, supersample=10)...")
    # --- Cropland fraction with AOI mask → NaN outside AOI, keep true 0 inside AOI
    cl_frac = fractional_rasterize_polygon(PATHS.CROPLAND, T, supersample=10)

    write_gtiff_masked(cl_frac, out_r("cropland_fraction_1km"), like=T, nodata=np.nan)
    mean_frac = float(np.nanmean(cl_frac.values))
    log.info(f"Wrote {out_r('cropland_fraction_1km').name} | mean_fraction={mean_frac:.3f}")

    # -------------------------------------------------------------------------
    # Electricity: grid / unelectrified masks (presence by cell)
    #   Field & codes (per user):
    #     - Use "FinalElecCode2020" where: 1 = Existing grid, 99 = Unelectrified
    #     - Fallback if Shapefile truncated: "FinalElecC"
    # -------------------------------------------------------------------------
    if PATHS.ELEC and Path(PATHS.ELEC).exists():
        elec_field = _pick_field(PATHS.ELEC, ["FinalElecCode2020", "FinalElecC"])
        if elec_field:
            log.info("Rasterizing electricity masks using field '%s' (1=grid, 99=unelectrified)...", elec_field)
            elc_grid = rasterize_vector(PATHS.ELEC, T, where=f"{elec_field} == 1", burn_value=1)
            elc_une  = rasterize_vector(PATHS.ELEC, T, where=f"{elec_field} == 99", burn_value=1)
            write_gtiff_masked(elc_grid, out_r("elec_grid_1km"), like=T, nodata=np.nan)
            write_gtiff_masked(elc_une,  out_r("elec_unelectrified_1km"), like=T, nodata=np.nan)
            log.info(f"Wrote {out_r('elec_grid_1km').name}, {out_r('elec_unelectrified_1km').name}")
        else:
            log.warning("Electricity layer present but required field not found; skipping elec rasters.")
    else:
        log.info("No electricity vector found; skipping elec rasters.")

    # -------------------------------------------------------------------------
    # Settlement type: urban / rural masks (presence by cell)
    #   Field & codes (per user):
    #     - Use "IsUrban" where: 2 = Urban, 0 = Rural
    # -------------------------------------------------------------------------
    if PATHS.SETTLE and Path(PATHS.SETTLE).exists():
        settle_field = _pick_field(PATHS.SETTLE, ["IsUrban"])
        if settle_field:
            log.info("Rasterizing settlement type using field '%s' (2=urban, 0=rural)...", settle_field)
            urb = rasterize_vector(PATHS.SETTLE, T, where=f"{settle_field} == 2", burn_value=1)
            rl  = rasterize_vector(PATHS.SETTLE, T, where=f"{settle_field} == 0", burn_value=1)
            write_gtiff_masked(urb, out_r("urban_1km"), like=T, nodata=np.nan)
            write_gtiff_masked(rl,  out_r("rural_1km"), like=T, nodata=np.nan)
            log.info(f"Wrote {out_r('urban_1km').name}, {out_r('rural_1km').name}")
        else:
            log.warning("Settlement layer present but 'IsUrban' not found; skipping settlement rasters.")
    else:
        log.info("No settlement vector found; skipping settlement rasters.")

    # -------------------------------------------------------------------------
    # Flood: keep native 30 m + write two 1-km products:
    #   (a) max depth per 1-km cell (screening)
    #   (b) coverage fraction per 1-km cell with depth ≥ FLOOD_DEPTH_RISK (0..1)
    # -------------------------------------------------------------------------
    log.info("Aggregating flood depth to 1-km products (max, and exceedance fraction)...")
    flood_30m = open_template(PATHS.FLOOD)

    # (a) 1-km MAX depth (did any subpixel get deep?) — good for quick visual checks
    flood_1km_max = flood_30m.rio.reproject_match(T, resampling=Resampling.max)
    write_gtiff_masked(flood_1km_max, out_r("flood_rp100_maxdepth_1km"), like=T, nodata=np.nan)
    log.info(f"Wrote {out_r('flood_rp100_maxdepth_1km').name}")

    # (b) 1-km EXCEEDANCE FRACTION:
    #     Convert native depth to a 0/1 exceedance mask, then average into 1-km cells.
    #     This answers “what share of the 1-km cell is flooded beyond the risk threshold?”
    thr_m = float(PARAMS.FLOOD_DEPTH_RISK)
    exceed_native = (flood_30m >= thr_m).astype("float32")

    # Average aggregation of a binary (0/1) mask gives the coverage fraction in [0,1]
    flood_1km_frac = exceed_native.rio.reproject_match(T, resampling=Resampling.average)
    write_gtiff_masked(flood_1km_frac, out_r("flood_rp100_exceed_frac_1km"), like=T, nodata=np.nan)

    # Quick diagnostic for logs
    with np.errstate(invalid="ignore"):
        mean_frac = float(np.nanmean(flood_1km_frac.values))
    log.info(
        "Wrote %s | threshold=%.2f m | mean_exceedance_fraction=%.3f",
        out_r("flood_rp100_exceed_frac_1km").name, thr_m, mean_frac
    )

    log.info("Step 00 complete.")


if __name__ == "__main__":
    main()

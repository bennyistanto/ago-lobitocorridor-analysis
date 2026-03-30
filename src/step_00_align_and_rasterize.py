"""
Step 00 — Align all base rasters to the travel-time grid and rasterize key vectors.

v2 — Major data upgrade (2026-03-29)
=====================================
This step now ingests ~30 input layers across multiple resolutions and
produces ~30+ aligned 1-km outputs.  The key design principle is
**multi-resolution zonal statistics**: high-resolution layers (100m
cropland, 500m phenology, ~464m NTL) are aggregated to 1-km cells via
appropriate statistics (fraction, mean, std) rather than naive bilinear
resampling.  Coarser layers (~4km SPEI) use bilinear interpolation.

New capabilities:
- MCD12Q2 phenology: 6 layers (GSL, EVI, NumCycles, greenup variability)
- SPEI-12 / SPI-3 drought: 9 layers (65-year run-theory products)
- VIIRS NTL time series: mean + trend (absolute and relative)
- Cropland from raster (100m ESA WorldCover) via zonal fraction
- Facility access: cost-distance to health/education via MAP friction surface
- Google Open Buildings: count, area, density per cell
- DRE Atlas settlements: population and energy demand per cell
- All legacy outputs preserved for backward compatibility

References:
  Malczewski & Rinner (2015). MCDA in GIS. Springer. [multi-resolution]
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling
import rioxarray as rxr

from config import (
    PATHS, PARAMS, out_r, get_logger,
    ADMIN2_ID_TIF, ADMIN2_LUT_CSV,
    # Canonical output names (v2)
    NTL_MEAN_1KM, NTL_TREND_1KM, NTL_TREND_PCT_1KM,
    GSL_MEDIAN_1KM, GSL_TREND_1KM, GREENUP_STDEV_1KM,
    EVI_AREA_1KM, EVI_AMPLITUDE_1KM, NUMCYCLES_1KM,
    SPEI12_NUM_EVENTS_1KM, SPEI12_MAX_DURATION_1KM,
    SPEI12_MEAN_INTENSITY_1KM, SPEI12_MIN_PEAK_1KM,
    SPEI12_MEAN_MAGNITUDE_1KM, SPEI12_NUM_EVENTS_RECENT_1KM,
    SPEI12_MAX_DURATION_RECENT_1KM, SPEI12_NUM_EVENTS_BASELINE_1KM,
    SPI03_NUM_EVENTS_1KM,
    CROPLAND_FRACTION_1KM,
    TT_HEALTH_MOTORISED_1KM, TT_HEALTH_WALKING_1KM,
    TT_EDUCATION_MOTORISED_1KM, TT_EDUCATION_WALKING_1KM,
    BUILDING_COUNT_1KM, BUILDING_AREA_1KM, BUILDING_DENSITY_1KM,
    DRE_POP_UNSERVED_1KM, DRE_DEMAND_DENSITY_1KM,
)

from utils_geo import (
    open_template,
    align_to_template,
    ensure_aligned,
    rasterize_vector,
    write_gtiff_masked,
    fractional_rasterize_polygon,
    align_rwi_to_template,
    rasterize_admin2_ids,
    apply_aoi_mask_if_enabled,
    # v2 functions
    zonal_stats_from_hires,
    cost_distance_to_nearest,
    point_density_in_cells,
    building_density_in_cells,
    dre_demand_in_cells,
)

warnings.filterwarnings("ignore", category=UserWarning)
log = get_logger(__name__)


def _pick_field(vec_path: Path, candidates: list[str]) -> str | None:
    """Return the first attribute name that exists in the vector file."""
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


def _align_optional(path, T, resampling, name, output_path):
    """Align an optional raster to template and write. Returns the aligned DA or None."""
    if path is None or not Path(path).exists():
        log.info("Skipping %s (not found).", name)
        return None
    try:
        da = align_to_template(path, T, resampling=resampling)
        # Handle -9999 nodata
        da = da.where(da != -9999)
        write_gtiff_masked(da, output_path, like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s", Path(output_path).name)
        return da
    except Exception as e:
        log.warning("Failed to align %s: %s", name, e)
        return None


def main() -> None:
    """
    Align inputs to the travel-time grid and produce AOI-prefixed base rasters.

    Sections:
      A — Core rasters (population, existing layers)
      B — Night-Time Lights (multi-year, trend)
      C — Phenology / Vegetation (MCD12Q2 derivatives)
      D — Drought / Climate (SPEI-12, SPI-3)
      E — Cropland (raster-based zonal fraction)
      F — Facility access (cost-distance via friction surfaces)
      G — Building density (Google Open Buildings)
      H — DRE settlement demand
      I — Existing vector layers (electricity, settlement, flood)
      J — Admin-2 labels
    """

    # =====================================================================
    # Template grid
    # =====================================================================
    T = open_template(PARAMS.TARGET_GRID)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        "Target grid | CRS=%s | size=%dx%d | cell=%.4f x %.4f",
        T.rio.crs, T.rio.height, T.rio.width, resx, resy
    )

    # =====================================================================
    # A — Core rasters: Population, RWI
    # =====================================================================
    log.info("=== Section A: Core rasters ===")

    pop_1km = align_to_template(PATHS.POP, T, resampling="nearest")
    write_gtiff_masked(pop_1km, out_r("pop_1km"), like=T, nodata=np.nan, force_mask=True)
    log.info("Wrote %s", out_r("pop_1km").name)

    # RWI (optional)
    try:
        if PATHS.RWI and Path(PATHS.RWI).exists():
            rwi_1km = align_rwi_to_template(PATHS.RWI, T)
            write_gtiff_masked(rwi_1km, out_r("rwi_meta_1km"), like=T, nodata=np.nan, force_mask=True)
            log.info("Wrote %s", out_r("rwi_meta_1km").name)
        else:
            log.info("No RWI file found; skipping.")
    except Exception as e:
        log.warning("RWI alignment skipped: %s", e)

    # =====================================================================
    # B — Night-Time Lights (VIIRS multi-year)
    # =====================================================================
    log.info("=== Section B: Night-Time Lights ===")

    # Primary: 10-year mean (replaces single-year)
    ntl_mean = _align_optional(PATHS.NTL_MEAN, T, "bilinear", "NTL mean 2015-2024", NTL_MEAN_1KM)

    # Backward compat: also write as ntl_1km for downstream steps
    if ntl_mean is not None:
        write_gtiff_masked(ntl_mean, out_r("ntl_1km"), like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s (backward compat alias)", out_r("ntl_1km").name)
    else:
        # Fallback to single-year 2024 if multi-year not available
        log.info("Falling back to single-year NTL 2024...")
        ntl_1km = align_to_template(PATHS.NTL, T, resampling="bilinear")
        write_gtiff_masked(ntl_1km, out_r("ntl_1km"), like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s (single-year fallback)", out_r("ntl_1km").name)

    # NTL trend layers
    _align_optional(PATHS.NTL_TREND_SLOPE, T, "bilinear", "NTL trend slope", NTL_TREND_1KM)
    _align_optional(PATHS.NTL_TREND_PCTYR, T, "bilinear", "NTL trend %/yr", NTL_TREND_PCT_1KM)

    # =====================================================================
    # C — Phenology / Vegetation (MCD12Q2)
    # =====================================================================
    log.info("=== Section C: Phenology (MCD12Q2) ===")

    # Primary: GSL median (replaces NDVI mean)
    gsl_med = _align_optional(PATHS.GSL_MEDIAN, T, "bilinear", "GSL median", GSL_MEDIAN_1KM)

    # Backward compat: also write as veg_1km
    if gsl_med is not None:
        write_gtiff_masked(gsl_med, out_r("veg_1km"), like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s (backward compat alias from GSL)", out_r("veg_1km").name)
    else:
        log.info("Falling back to NDVI mean 2024...")
        veg_1km = align_to_template(PATHS.VEG, T, resampling="bilinear")
        write_gtiff_masked(veg_1km, out_r("veg_1km"), like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s (NDVI fallback)", out_r("veg_1km").name)

    # Supplementary phenology layers
    _align_optional(PATHS.GSL_TREND, T, "bilinear", "GSL trend", GSL_TREND_1KM)
    _align_optional(PATHS.GREENUP_STDEV, T, "bilinear", "Greenup StdDev", GREENUP_STDEV_1KM)
    _align_optional(PATHS.EVI_AREA, T, "bilinear", "EVI Area", EVI_AREA_1KM)
    _align_optional(PATHS.EVI_AMPLITUDE, T, "bilinear", "EVI Amplitude", EVI_AMPLITUDE_1KM)
    _align_optional(PATHS.NUMCYCLES, T, "nearest", "NumCycles", NUMCYCLES_1KM)  # categorical → nearest

    # =====================================================================
    # D — Drought / Climate (SPEI-12, SPI-3)
    # =====================================================================
    log.info("=== Section D: Drought (SPEI-12 / SPI-3) ===")

    # Primary: SPEI-12 num_events (replaces FAO ASI)
    spei_ne = _align_optional(PATHS.SPEI12_NUM_EVENTS, T, "bilinear",
                               "SPEI-12 num_events", SPEI12_NUM_EVENTS_1KM)

    # Backward compat: also write as drought_1km
    if spei_ne is not None:
        write_gtiff_masked(spei_ne, out_r("drought_1km"), like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s (backward compat alias from SPEI)", out_r("drought_1km").name)
    else:
        log.info("Falling back to FAO ASI 2024...")
        drt_1km = align_to_template(PATHS.DROUGHT, T, resampling="bilinear")
        write_gtiff_masked(drt_1km, out_r("drought_1km"), like=T, nodata=np.nan, force_mask=True)
        log.info("Wrote %s (FAO ASI fallback)", out_r("drought_1km").name)

    # Supplementary SPEI layers
    _align_optional(PATHS.SPEI12_MAX_DURATION, T, "bilinear",
                    "SPEI-12 max_duration", SPEI12_MAX_DURATION_1KM)
    _align_optional(PATHS.SPEI12_MEAN_INTENSITY, T, "bilinear",
                    "SPEI-12 mean_intensity", SPEI12_MEAN_INTENSITY_1KM)
    _align_optional(PATHS.SPEI12_MIN_PEAK, T, "bilinear",
                    "SPEI-12 min_peak", SPEI12_MIN_PEAK_1KM)
    _align_optional(PATHS.SPEI12_MEAN_MAGNITUDE, T, "bilinear",
                    "SPEI-12 mean_magnitude", SPEI12_MEAN_MAGNITUDE_1KM)
    _align_optional(PATHS.SPEI12_NUM_EVENTS_RECENT, T, "bilinear",
                    "SPEI-12 num_events recent", SPEI12_NUM_EVENTS_RECENT_1KM)
    _align_optional(PATHS.SPEI12_MAX_DURATION_RECENT, T, "bilinear",
                    "SPEI-12 max_duration recent", SPEI12_MAX_DURATION_RECENT_1KM)
    _align_optional(PATHS.SPEI12_NUM_EVENTS_BASELINE, T, "bilinear",
                    "SPEI-12 num_events baseline", SPEI12_NUM_EVENTS_BASELINE_1KM)
    _align_optional(PATHS.SPI03_NUM_EVENTS, T, "bilinear",
                    "SPI-3 num_events", SPI03_NUM_EVENTS_1KM)

    # =====================================================================
    # E — Cropland (raster-based zonal fraction, replaces vector supersample)
    # =====================================================================
    log.info("=== Section E: Cropland ===")

    if PATHS.CROPLAND_RASTER and Path(PATHS.CROPLAND_RASTER).exists():
        log.info("Computing cropland fraction from 100m raster (zonal stats)...")
        crop_stats = zonal_stats_from_hires(
            PATHS.CROPLAND_RASTER, T,
            stats=("fraction",),
            nodata=-9999,
        )
        if "fraction" in crop_stats:
            cl_frac = crop_stats["fraction"]
            write_gtiff_masked(cl_frac, CROPLAND_FRACTION_1KM, like=T, nodata=np.nan, force_mask=True)
            # Also write as backward-compat name
            write_gtiff_masked(cl_frac, out_r("cropland_fraction_1km"), like=T, nodata=np.nan, force_mask=True)
            mean_frac = float(np.nanmean(cl_frac.values))
            log.info("Wrote cropland_fraction_1km | mean=%.3f (from 100m raster)", mean_frac)

            # Presence mask (any cropland in cell)
            cl_pres = xr.where(cl_frac > 0, 1.0, 0.0).astype("float32")
            cl_pres = cl_pres.rio.write_crs(T.rio.crs)
            cl_pres = cl_pres.rio.write_transform(T.rio.transform())
            write_gtiff_masked(cl_pres, out_r("cropland_presence_1km"), like=T, nodata=np.nan, force_mask=True)
            log.info("Wrote cropland_presence_1km (from raster)")
        else:
            log.warning("Cropland zonal stats returned no 'fraction'. Falling back to vector.")
            _cropland_from_vector(T)
    else:
        log.info("No cropland raster found; falling back to vector rasterisation.")
        _cropland_from_vector(T)

    # =====================================================================
    # F — Facility access via MAP friction surfaces + MCP cost-distance
    # =====================================================================
    # MAP friction rasters are in minutes/metre.  For each (facility type,
    # travel mode) pair we run MCP cost-distance from facility points over
    # the friction surface to produce travel-time-to-nearest rasters.
    log.info("=== Section F: Facility access (MCP cost-distance) ===")

    def _cost_distance_and_write(friction_path, facility_path, template,
                                 out_path, label, max_minutes=360.0):
        """Run MCP cost-distance from facility points over friction surface."""
        if friction_path is None or not Path(friction_path).exists():
            log.info("Skipping %s (friction raster not found).", label)
            return
        if facility_path is None or not Path(facility_path).exists():
            log.info("Skipping %s (facility points not found).", label)
            return
        log.info("Computing cost-distance for %s...", label)
        try:
            fac_gdf = gpd.read_file(facility_path)
            result = cost_distance_to_nearest(
                friction_path, fac_gdf, template, max_minutes=max_minutes)
            if result is not None:
                write_gtiff_masked(result, out_path, like=template,
                                   nodata=np.nan, force_mask=True)
                log.info("Wrote %s", Path(out_path).name)
            else:
                log.warning("%s cost-distance returned None.", label)
        except Exception as e:
            log.warning("%s cost-distance failed: %s", label, e)

    # Health — motorised
    _cost_distance_and_write(
        PATHS.FRICTION_MOTORISED, PATHS.HEALTH_FACILITIES,
        T, TT_HEALTH_MOTORISED_1KM, "health (motorised)", max_minutes=360.0)

    # Health — walking
    _cost_distance_and_write(
        PATHS.FRICTION_WALKING, PATHS.HEALTH_FACILITIES,
        T, TT_HEALTH_WALKING_1KM, "health (walking)", max_minutes=720.0)

    # Education — motorised
    _cost_distance_and_write(
        PATHS.FRICTION_MOTORISED, PATHS.EDUCATION_FACILITIES,
        T, TT_EDUCATION_MOTORISED_1KM, "education (motorised)", max_minutes=360.0)

    # Education — walking
    _cost_distance_and_write(
        PATHS.FRICTION_WALKING, PATHS.EDUCATION_FACILITIES,
        T, TT_EDUCATION_WALKING_1KM, "education (walking)", max_minutes=720.0)

    # =====================================================================
    # G — Building density (Google Open Buildings)
    # =====================================================================
    log.info("=== Section G: Building density ===")

    if PATHS.OPEN_BUILDINGS and Path(PATHS.OPEN_BUILDINGS).exists():
        log.info("Computing building density from Open Buildings...")
        try:
            bldg_gdf = gpd.read_file(PATHS.OPEN_BUILDINGS)
            bldg_stats = building_density_in_cells(bldg_gdf, T, area_col="area_in_me")

            for key, out_path in [("count", BUILDING_COUNT_1KM),
                                  ("area", BUILDING_AREA_1KM),
                                  ("density", BUILDING_DENSITY_1KM)]:
                if key in bldg_stats:
                    write_gtiff_masked(bldg_stats[key], out_path, like=T, nodata=np.nan, force_mask=True)
                    log.info("Wrote %s", Path(out_path).name)

            n_buildings = int(len(bldg_gdf))
            log.info("Processed %d buildings", n_buildings)
        except Exception as e:
            log.warning("Building density computation failed: %s", e)
    else:
        log.info("No Open Buildings data found; skipping.")

    # =====================================================================
    # H — DRE settlement demand
    # =====================================================================
    log.info("=== Section H: DRE settlement demand ===")

    if PATHS.DRE_SETTLEMENTS and Path(PATHS.DRE_SETTLEMENTS).exists():
        log.info("Computing DRE demand from settlement polygons...")
        try:
            dre_gdf = gpd.read_file(PATHS.DRE_SETTLEMENTS)
            dre_stats = dre_demand_in_cells(dre_gdf, T,
                                            pop_col="population",
                                            demand_col="demand")
            if "population" in dre_stats:
                write_gtiff_masked(dre_stats["population"], DRE_POP_UNSERVED_1KM,
                                   like=T, nodata=np.nan, force_mask=True)
                log.info("Wrote %s", Path(DRE_POP_UNSERVED_1KM).name)
            if "demand" in dre_stats:
                write_gtiff_masked(dre_stats["demand"], DRE_DEMAND_DENSITY_1KM,
                                   like=T, nodata=np.nan, force_mask=True)
                log.info("Wrote %s", Path(DRE_DEMAND_DENSITY_1KM).name)

            n_settlements = int(len(dre_gdf))
            log.info("Processed %d DRE settlements", n_settlements)
        except Exception as e:
            log.warning("DRE demand computation failed: %s", e)
    else:
        log.info("No DRE settlement data found; skipping.")

    # =====================================================================
    # I — Existing vector layers (electricity, settlement, flood)
    # =====================================================================
    log.info("=== Section I: Existing vector/raster layers ===")

    # Electricity
    if PATHS.ELEC and Path(PATHS.ELEC).exists():
        elec_field = _pick_field(PATHS.ELEC, ["FinalElecCode2020", "FinalElecC"])
        if elec_field:
            log.info("Rasterizing electricity masks using field '%s'...", elec_field)
            elc_grid = rasterize_vector(PATHS.ELEC, T, where=f"{elec_field} == 1", burn_value=1)
            elc_une = rasterize_vector(PATHS.ELEC, T, where=f"{elec_field} == 99", burn_value=1)
            write_gtiff_masked(elc_grid, out_r("elec_grid_1km"), like=T, nodata=np.nan, force_mask=True)
            write_gtiff_masked(elc_une, out_r("elec_unelectrified_1km"), like=T, nodata=np.nan, force_mask=True)
            log.info("Wrote elec_grid_1km, elec_unelectrified_1km")
    else:
        log.info("No electricity vector found; skipping.")

    # Settlement type
    if PATHS.SETTLE and Path(PATHS.SETTLE).exists():
        settle_field = _pick_field(PATHS.SETTLE, ["IsUrban"])
        if settle_field:
            log.info("Rasterizing settlement type using field '%s'...", settle_field)
            urb = rasterize_vector(PATHS.SETTLE, T, where=f"{settle_field} == 2", burn_value=1)
            rl = rasterize_vector(PATHS.SETTLE, T, where=f"{settle_field} == 0", burn_value=1)
            write_gtiff_masked(urb, out_r("urban_1km"), like=T, nodata=np.nan, force_mask=True)
            write_gtiff_masked(rl, out_r("rural_1km"), like=T, nodata=np.nan, force_mask=True)
            log.info("Wrote urban_1km, rural_1km")
    else:
        log.info("No settlement vector found; skipping.")

    # Flood
    if PATHS.FLOOD and Path(PATHS.FLOOD).exists():
        log.info("Aggregating flood depth to 1-km products...")
        flood_30m = open_template(PATHS.FLOOD)

        flood_1km_max = flood_30m.rio.reproject_match(T, resampling=Resampling.max)
        write_gtiff_masked(flood_1km_max, out_r("flood_rp100_maxdepth_1km"), like=T, nodata=np.nan, force_mask=True)

        thr_m = float(PARAMS.FLOOD_DEPTH_RISK)
        exceed_native = (flood_30m >= thr_m).astype("float32")
        flood_1km_frac = exceed_native.rio.reproject_match(T, resampling=Resampling.average)
        write_gtiff_masked(flood_1km_frac, out_r("flood_rp100_exceed_frac_1km"), like=T, nodata=np.nan, force_mask=True)

        with np.errstate(invalid="ignore"):
            mean_frac = float(np.nanmean(flood_1km_frac.values))
        log.info("Wrote flood products | threshold=%.2f m | mean_exceed=%.3f", thr_m, mean_frac)
    else:
        log.info("No flood raster found; skipping.")

    # =====================================================================
    # J — Admin-2 labels
    # =====================================================================
    log.info("=== Section J: Admin-2 labels ===")

    try:
        adm2_path = PATHS.BND_ADM2
        if adm2_path and Path(adm2_path).exists():
            gdf = gpd.read_file(adm2_path)
            try:
                gdf = gdf.to_crs(T.rio.crs)
            except Exception:
                gdf.set_crs(T.rio.crs, inplace=True)

            idgrid_da, lut_df = rasterize_admin2_ids(
                gdf=gdf, template=T,
                code_field="ADM2CD_c", name1_field="NAM_1", name2_field="NAM_2",
                all_touched=False,
            )
            idgrid_float = idgrid_da.astype("float32")
            write_gtiff_masked(idgrid_float, ADMIN2_ID_TIF, like=T, nodata=np.nan, force_mask=True)
            lut_df.to_csv(ADMIN2_LUT_CSV, index=False)
            log.info("Wrote %s, %s", Path(ADMIN2_ID_TIF).name, Path(ADMIN2_LUT_CSV).name)
        else:
            log.info("Admin-2 boundary not found; skipping.")
    except Exception as e:
        log.warning("Admin-2 rasterization skipped: %s", e)

    log.info("Step 00 complete.")


def _cropland_from_vector(T):
    """Legacy cropland rasterization from vector (fallback if raster not available)."""
    log.info("Rasterizing cropland from vector (legacy fallback)...")
    cl_pres = rasterize_vector(PATHS.CROPLAND, T, burn_value=1)
    write_gtiff_masked(cl_pres, out_r("cropland_presence_1km"), like=T, nodata=np.nan, force_mask=True)

    cl_frac = fractional_rasterize_polygon(PATHS.CROPLAND, T, supersample=10)
    write_gtiff_masked(cl_frac, out_r("cropland_fraction_1km"), like=T, nodata=np.nan, force_mask=True)
    cl_frac_masked = apply_aoi_mask_if_enabled(cl_frac, T, force_mask=True)
    mean_frac = float(np.nanmean(cl_frac_masked.values))
    log.info("Wrote cropland_fraction_1km | mean=%.3f (from vector fallback)", mean_frac)


if __name__ == "__main__":
    main()

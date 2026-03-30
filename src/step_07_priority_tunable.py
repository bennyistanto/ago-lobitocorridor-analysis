"""
Step 07 — Tunable Priority Surface (raster-only)

Purpose
-------
Compute a policy-facing priority surface with easy switches & masks:
- Components (ACCESS, POP, VEG, NTL, DROUGHT) are toggleable via config.USE_COMPONENTS.
- Optional overlays from Step 06 and Step 00 (if present & aligned):
  * Admin2 poverty (0..1), food insecurity (0..1), muni travel time (minutes)
  * RWI (Meta) from Step 00 aligned product; scaled -2..2 → 0..1 then inverted so poorer → higher priority
- Rural-only and minimum cropland masks are supported.
- Light smoothing (focal mean) is available before thresholding to grow coherent patches.
- Selection can be by Top % of valid cells or fixed Top km² (area-true).
- Tiny speckle clusters can be removed by size.

Inputs (expected from earlier steps)
------------------------------------
- Step 00 outputs:
  * {AOI}_pop_1km.tif, {AOI}_veg_1km.tif, {AOI}_ntl_1km.tif, {AOI}_drought_1km.tif
  * {AOI}_cropland_fraction_1km.tif, {AOI}_rural_1km.tif
  * (optional) {AOI}_rwi_meta_1km.tif
  * (Template grid is PARAMS.TARGET_GRID, usually the travel-time raster)
- Optional Step 06 outputs (if present):
  * {AOI}_muni_poverty_poverty_rural_1km.tif
  * {AOI}_muni_foodinsecurity_food_insec_scale_1km.tif
  * {AOI}_muni_traveltime_avg_hours_to_market_financial_1km.tif  (NOTE: Step 06 writes minutes; name kept for compatibility)

Config knobs (from config.PARAMS)
---------------------------------
- USE_COMPONENTS: (ACC, POP, VEG, NTL, DRT) → 1/0 to include/exclude
- Legacy weights: W_ACC, W_POP, W_VEG, W_NTL, W_DRT (re-normalized over included comps)
- Optional overlay weights (if present; defaults below if missing):
  * W_POV (default 0.15), W_FOOD (0.10), W_MTT (0.10), W_RWI (0.15)
- Masks: MASK_REQUIRE_RURAL (bool), MASK_MIN_CROPLAND (float, 0 disables)
- Caps: NTL_CAP (e.g., 0.20), VEG_MIN (e.g., 0.40)
- Smoothing/clusters: SMOOTH_RADIUS (0/1/2), MIN_CLUSTER_CELLS
- Selection: TOP_PCT_CELLS or TOP_KM2 (set only one)

Outputs
-------
- {AOI}_priority_score_0_1.tif          (continuous 0..1)
- {AOI}_priority_top10_mask.tif         (1=selected; name kept for compatibility even if Top-km² used)
"""


from __future__ import annotations
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Dict
from rasterio.enums import Resampling
import pandas as pd
import rioxarray as rxr

from config import (
    AOI, PATHS, PARAMS, Params, TransformSpec,
    out_r, get_logger,
    PRIORITY_TIF, PRIORITY_TOP10_TIF,
    OPTIONAL_GRID_OVERLAYS,
    RESAMPLE_DEFAULT_CONT, RESAMPLE_DEFAULT_CAT,
    WRITE_JSON_SIDECARS, write_geo_sidecar,
    PRESETS, ACTIVE_PRESET, preset_to_params,
)
# Safe fallbacks if these constants aren’t present in config
try:
    from config import ADMIN2_ID_TIF, ADMIN2_LUT_CSV
except Exception:
    from config import out_t
    ADMIN2_ID_TIF = out_r("admin2_id_1km")
    ADMIN2_LUT_CSV = out_t("admin2_lookup")

from utils_geo import (
    open_template, write_gtiff_masked,
    focal_mean, cell_area_km2_latlon,
    apply_aoi_mask_if_enabled,
    select_top_mask_nan as select_top,
    remove_small_clusters as prune_clusters,
    ensure_aligned, open_and_align,
    # v2 analytical functions
    fuzzy_transform,
    geometric_aggregate,
    getis_ord_gi_star,
)

log = get_logger(__name__)


# ------------------------------ IO helpers -----------------------------------

# ---- Admin-2 rank schema lock (13 columns) ----
EXPECTED_RANK_COLS = [
    "ADM2CD_c", "NAM_1", "NAM_2",
    "score", "rank",
    "selected",
    "share_selected",
    "selected_cells", "selected_km2",
    "total_cells", "total_km2",
    "top10_cells", "top10_km2",
]

def _ensure_rank_columns(df):
    """
    Enforce expected columns, defaults, order, and dtypes.
    Returns a new DataFrame containing only EXPECTED_RANK_COLS.
    """
    import numpy as np
    import pandas as pd

    out = df.copy()

    # Defaults if missing
    for c in EXPECTED_RANK_COLS:
        if c not in out.columns:
            if c == "selected":
                out[c] = False
            elif c in ("selected_cells", "total_cells", "top10_cells", "rank"):
                out[c] = 0
            elif c in ("selected_km2", "total_km2", "top10_km2", "share_selected", "score"):
                out[c] = 0.0
            else:
                out[c] = np.nan

    # Dtypes
    out["selected"] = out["selected"].astype(bool)
    for c in ("selected_cells", "total_cells", "top10_cells", "rank"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    for c in ("selected_km2", "total_km2", "top10_km2", "share_selected", "score"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    # Keep only the expected columns, in order
    return out[EXPECTED_RANK_COLS]


def _r(path: str) -> xr.DataArray | None:
    """Open a raster if it exists; return None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    da = xr.open_dataarray(p)
    # ensure rioxarray attrs are present (xarray open_dataarray may not load them fully)
    try:
        _ = da.rio.crs
    except Exception:
        # open via rioxarray if needed
        import rioxarray as rxr
        da = rxr.open_rasterio(p, masked=True).squeeze()
    return da


def _find_optional_overlays(T: xr.DataArray) -> dict[str, xr.DataArray]:
    """Open any optional overlays declared in config. Reproject-match if needed."""
    overlays: dict[str, xr.DataArray] = {}

    for alias, base in (OPTIONAL_GRID_OVERLAYS or {}).items():
        p = out_r(base)          # base WITHOUT .tif (writer appends)
        p_tif = p.with_suffix(".tif")
        if not p_tif.exists():
            log.info("Optional overlay not found: %s", p_tif.name)
            continue

        try:
            da = xr.open_dataarray(p_tif)
        except Exception as e_xr:
            try:
                da = rxr.open_rasterio(p_tif, masked=True).squeeze()
            except Exception as e_rio:
                log.warning(
                    "Failed to open overlay %s (xarray: %s; rioxarray: %s). Skipping.",
                    p_tif.name, e_xr, e_rio,
                )
                continue

        da = ensure_aligned(da, T, resampling=RESAMPLE_DEFAULT_CONT)
        if da is not None:
            overlays[alias] = da
    return overlays


# ------------------------------ Scoring utils --------------------------------

def _safe_minmax_scale(da: xr.DataArray, lo: float, hi: float, invert: bool = False) -> xr.DataArray:
    """
    Clip to [lo,hi] then scale to [0,1]. If invert=True, return 1 - scaled.
    """
    out = da.clip(lo, hi)
    out = (out - lo) / (hi - lo + 1e-9)
    if invert:
        out = 1.0 - out
    return out


def _normalize_components(T, tt_da, pop_da, veg_da, ntl_da, drt_da, overlays: Dict[str, xr.DataArray], params: Params = PARAMS) -> Dict[str, xr.DataArray]:
    """
    Produce normalized [0,1] components: ACC, POP, VEG, NTL, DRT (+ optional overlays).
    - ACCESS: lower travel is better → invert after min-max (0..max_iso)
    - POP: robust scaling using 95th percentile to avoid extreme skew
    - VEG: threshold at VEG_MIN (values below → 0), then linear to 1
    - NTL: cap at NTL_CAP then linear to 1
    - DRT: drought frequency (0..30%) → invert (less drought is better).
           If raster looks like 0..100, auto-scale to 0..1 first.
    - poverty (optional): already 0..1 from Step 06 (higher → higher priority)
    - food (optional): already 0..1 from Step 06 (higher → higher priority)
    - muni_tt (optional): minutes; lower is better (invert)
    - rwi (optional): Meta RWI scaled -2..2 → 0..1 then inverted (poorer → higher priority)
    """
    comps = {}

    # Access
    max_iso = float(max(params.ISO_THRESH))  # e.g., 240
    comps["ACC"] = _safe_minmax_scale(tt_da, 0.0, max_iso, invert=True)

    # Population (95th pct robust scaling)
    if pop_da is not None:
        v = pop_da.values.astype("float32")
        v[v < 0] = np.nan
        p95 = np.nanpercentile(v, 95.0)
        p95 = p95 if np.isfinite(p95) and p95 > 0 else (np.nanmax(v) or 1.0)
        comps["POP"] = _safe_minmax_scale(pop_da, 0.0, float(p95), invert=False).clip(0, 1)
    else:
        comps["POP"] = None

    # Vegetation
    if veg_da is not None:
        veg_min = float(params.VEG_MIN)
        # map < VEG_MIN to 0; else linear to 1
        vv = veg_da.clip(veg_min, 1.0)
        comps["VEG"] = _safe_minmax_scale(vv, veg_min, 1.0, invert=False)
    else:
        comps["VEG"] = None

    # Night Lights
    if ntl_da is not None:
        ntl_cap = float(params.NTL_CAP)
        nn = ntl_da.clip(0.0, ntl_cap)
        comps["NTL"] = _safe_minmax_scale(nn, 0.0, ntl_cap, invert=False)
    else:
        comps["NTL"] = None

    # Drought (auto-detect scale)
    if drt_da is not None:
        # try to detect if it's 0..100 (percent) or 0..1
        med = float(np.nanmedian(drt_da.values))
        if med > 1.0:
            dr = drt_da / 100.0
        else:
            dr = drt_da
        # cap at 0.30 (30%)
        drc = dr.clip(0.0, 0.30)
        comps["DRT"] = _safe_minmax_scale(drc, 0.0, 0.30, invert=True)
    else:
        comps["DRT"] = None

    # Optional overlays
    if "poverty" in overlays:
        comps["POV"] = overlays["poverty"].clip(0.0, 1.0)
    if "food" in overlays:
        comps["FOOD"] = overlays["food"].clip(0.0, 1.0)
    if "muni_tt" in overlays:
        # Step 06 writes minutes, but protect scaling if hours slipped in
        mtt = overlays["muni_tt"]
        m_med = float(np.nanmedian(mtt.values))
        # Heuristic: if median < 6, treat as hours and convert to minutes
        if m_med < 6.0:
            log.info("muni_tt appears to be in hours (median < 6). Converting to minutes.")
            mtt = mtt * 60.0
        comps["MTT"] = _safe_minmax_scale(mtt, 0.0, max_iso, invert=True)
    # RWI: (-2..2) → scale to 0..1 using robust min/max; then INVERT so poorer → higher priority
    if "rwi" in overlays:
        r = overlays["rwi"].astype("float32")
        has_vals = np.isfinite(r.values).any()
        rmin = np.nanpercentile(r.values, 5) if has_vals else -2.0
        rmax = np.nanpercentile(r.values, 95) if has_vals else  2.0
        r = r.clip(rmin, rmax)
        scaled = (r - rmin) / (rmax - rmin + 1e-9)
        comps["RWI"] = 1.0 - scaled

    return comps


def _combine_with_weights(comps: Dict[str, xr.DataArray], params: Params = PARAMS) -> tuple[xr.DataArray, Dict[str, float]]:
    """
    Combine enabled components with normalized weights.

    - Core toggles follow params.USE_COMPONENTS (ACC, POP, VEG, NTL, DRT).
    - Optional overlays are included only if present in `comps`:
      POV (poverty), FOOD (food insecurity), MTT (muni travel time), RWI (Meta).
    - Weights are taken from params and re-normalized over the actually-available set.
    Returns:
      (score_da_clipped_0_1, weights_normalized_dict)
    """
    # Build available components + weights
    weights: Dict[str, float] = {}

    use_acc, use_pop, use_veg, use_ntl, use_drt = params.USE_COMPONENTS

    if use_acc and (comps.get("ACC") is not None):
        weights["ACC"] = float(params.W_ACC)
    if use_pop and (comps.get("POP") is not None):
        weights["POP"] = float(params.W_POP)
    if use_veg and (comps.get("VEG") is not None):
        weights["VEG"] = float(params.W_VEG)
    if use_ntl and (comps.get("NTL") is not None):
        weights["NTL"] = float(params.W_NTL)
    if use_drt and (comps.get("DRT") is not None):
        weights["DRT"] = float(params.W_DRT)

    # Optional municipal overlays — include only if present
    if "POV" in comps:
        weights["POV"] = float(getattr(params, "W_POV", 0.0))
    if "FOOD" in comps:
        weights["FOOD"] = float(getattr(params, "W_FOOD", 0.0))
    if "MTT" in comps:
        weights["MTT"] = float(getattr(params, "W_MTT", 0.0))
    if "RWI" in comps:
        weights["RWI"] = float(getattr(params, "W_RWI", 0.0))

    # Keep only positive weights and normalize
    weights = {k: v for k, v in weights.items() if v is not None and v > 0}
    w_sum = sum(weights.values())
    if w_sum <= 0:
        raise RuntimeError("No positive weights for any available component.")

    weights_norm = {k: v / w_sum for k, v in weights.items()}

    # Log effective blend
    log.info("Priority weight blend → " + ", ".join(f"{k}:{weights_norm[k]:.2f}" for k in weights_norm))

    # Weighted linear blend
    score: xr.DataArray | None = None
    for key, w in weights_norm.items():
        da = comps[key]
        score = da * w if score is None else score + da * w

    return score.clip(0.0, 1.0), weights_norm


def _apply_masks(score: xr.DataArray, T, rural_da, cropfrac_da, params: Params = PARAMS) -> xr.DataArray:
    """
    Apply rural-only and minimum cropland fraction masks if requested in config.
    """
    out = score.copy()
    if bool(params.MASK_REQUIRE_RURAL):
        if rural_da is None:
            log.warning("MASK_REQUIRE_RURAL=True but no rural raster found; skipping this mask.")
        else:
            out = out.where(rural_da > 0.5)
    min_cf = float(params.MASK_MIN_CROPLAND or 0.0)
    if min_cf > 0.0:
        if cropfrac_da is None:
            log.warning("MASK_MIN_CROPLAND>0 but no cropland_fraction raster found; skipping this mask.")
        else:
            out = out.where(cropfrac_da >= min_cf)
    return out


def _smooth_if_needed(score: xr.DataArray, params: Params = PARAMS) -> xr.DataArray:
    r = int(params.SMOOTH_RADIUS or 0)
    if r <= 0:
        return score
    return focal_mean(score, radius=r)



def _add_v2_components(
    comps: Dict[str, xr.DataArray],
    tt_health_da, tt_edu_da, bldg_dens_da, dre_demand_da,
    transforms: dict | None,
    params: Params,
) -> None:
    """Add v2 component layers to the comps dict.

    For layers WITHOUT a custom TransformSpec, apply simple linear
    normalisation.  When a preset specifies a TransformSpec for the
    alias, the raw (un-normalised) value is stored — the main loop
    applies the fuzzy transform later.
    """
    # Travel time to health — if no custom transform, linear invert
    if tt_health_da is not None:
        if transforms and "TT_HEALTH" in transforms:
            comps["TT_HEALTH"] = tt_health_da  # raw — will be transformed later
        else:
            max_tt = float(max(params.ISO_THRESH))
            comps["TT_HEALTH"] = _safe_minmax_scale(tt_health_da, 0.0, max_tt, invert=True)

    # Travel time to education — same pattern
    if tt_edu_da is not None:
        if transforms and "TT_EDUCATION" in transforms:
            comps["TT_EDUCATION"] = tt_edu_da
        else:
            max_tt = float(max(params.ISO_THRESH))
            comps["TT_EDUCATION"] = _safe_minmax_scale(tt_edu_da, 0.0, max_tt, invert=True)

    # Building density — log scale or raw for transform
    if bldg_dens_da is not None:
        if transforms and "BUILDING_DENSITY" in transforms:
            comps["BUILDING_DENSITY"] = bldg_dens_da
        else:
            v = bldg_dens_da.values.astype("float32")
            v[v < 0] = np.nan
            p95 = np.nanpercentile(v, 95.0) if np.any(np.isfinite(v)) else 1.0
            p95 = p95 if np.isfinite(p95) and p95 > 0 else 1.0
            comps["BUILDING_DENSITY"] = _safe_minmax_scale(bldg_dens_da, 0.0, float(p95))

    # DRE demand — linear scale or raw for transform
    if dre_demand_da is not None:
        if transforms and "DRE_DEMAND" in transforms:
            comps["DRE_DEMAND"] = dre_demand_da
        else:
            v = dre_demand_da.values.astype("float32")
            v[v < 0] = np.nan
            p95 = np.nanpercentile(v, 95.0) if np.any(np.isfinite(v)) else 1.0
            p95 = p95 if np.isfinite(p95) and p95 > 0 else 1.0
            comps["DRE_DEMAND"] = _safe_minmax_scale(dre_demand_da, 0.0, float(p95))


def _build_weight_dict(comps: Dict[str, xr.DataArray], params: Params) -> Dict[str, float]:
    """Build a complete weight dictionary for all available components."""
    weights: Dict[str, float] = {}

    use_acc, use_pop, use_veg, use_ntl, use_drt = params.USE_COMPONENTS

    if use_acc and comps.get("ACC") is not None:
        weights["ACC"] = float(params.W_ACC)
    if use_pop and comps.get("POP") is not None:
        weights["POP"] = float(params.W_POP)
    if use_veg and comps.get("VEG") is not None:
        weights["VEG"] = float(params.W_VEG)
    if use_ntl and comps.get("NTL") is not None:
        weights["NTL"] = float(params.W_NTL)
    if use_drt and comps.get("DRT") is not None:
        weights["DRT"] = float(params.W_DRT)

    # Optional overlays
    for alias, attr in [("POV", "W_POV"), ("FOOD", "W_FOOD"),
                        ("MTT", "W_MTT"), ("RWI", "W_RWI")]:
        if alias in comps and comps[alias] is not None:
            w = float(getattr(params, attr, 0.0))
            if w > 0:
                weights[alias] = w

    # v2 components
    for alias, attr in [("TT_HEALTH", "W_TT_HEALTH"),
                        ("TT_EDUCATION", "W_TT_EDUCATION"),
                        ("BUILDING_DENSITY", "W_BUILDING_DENSITY"),
                        ("DRE_DEMAND", "W_DRE_DEMAND")]:
        if alias in comps and comps[alias] is not None:
            w = float(getattr(params, attr, 0.0))
            if w > 0:
                weights[alias] = w

    return {k: v for k, v in weights.items() if v > 0}


def _admin2_rank_path():
    # One consistent file name for the required table
    from config import out_t as _out_t
    return _out_t("priority_admin2_rank")


# --------------------------------- Main --------------------------------------

def main(params: Params | None = None) -> None:
    """
    Compute a tunable priority surface with masks, smoothing, and Top-X selection.
    Produces:
      - PRIORITY_TIF (continuous 0..1)
      - PRIORITY_TOP10_TIF (binary mask, even if Top-km² is used)

    Parameters
    ----------
    params : Params | None
        Override the global PARAMS.  When *None*, resolved from the active
        thematic preset (``ACTIVE_PRESET`` env var, default ``"balanced"``).
    """
    if params is None:
        preset = PRESETS.get(ACTIVE_PRESET, PRESETS["balanced"])
        params = preset_to_params(preset)
        log.info("Preset: %s — %s", preset.name, preset.description)
    else:
        preset = PRESETS.get(ACTIVE_PRESET, PRESETS["balanced"])

    # Template (1-km travel grid) for shape/transform/CRS
    T = open_template(params.TARGET_GRID)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info("Target grid | CRS=%s | size=%dx%d | cell=%.4f x %.4f",
             T.rio.crs, T.rio.height, T.rio.width, resx, resy)

    # ── Load all component layers ────────────────────────────────────
    tt_da   = xr.open_dataarray(params.TARGET_GRID)  # travel time (minutes)
    pop_da  = open_and_align(out_r("pop_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    veg_da  = open_and_align(out_r("veg_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    ntl_da  = open_and_align(out_r("ntl_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    drt_da  = open_and_align(out_r("drought_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    crop_da = open_and_align(out_r("cropland_fraction_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    rur_da  = open_and_align(out_r("rural_1km"), T, resampling=RESAMPLE_DEFAULT_CAT)

    # v2 layers (optional — loaded only if Step 00 produced them)
    tt_health_da   = open_and_align(out_r("tt_health_motorised_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    tt_edu_da      = open_and_align(out_r("tt_education_motorised_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    bldg_dens_da   = open_and_align(out_r("building_density_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)
    dre_demand_da  = open_and_align(out_r("dre_demand_density_1km"), T, resampling=RESAMPLE_DEFAULT_CONT)

    # Optional overlays (Step 06 + Step 00 RWI)
    overlays = _find_optional_overlays(T)

    # ── Determine methodology from preset ────────────────────────────
    aggregation = getattr(preset, "AGGREGATION", "additive")
    selection_method = getattr(preset, "SELECTION_METHOD", "percentile")
    transforms = getattr(preset, "TRANSFORMS", None)
    knockout_rules = getattr(preset, "KNOCKOUT_RULES", None)
    hotspot_sig = getattr(preset, "HOTSPOT_SIGNIFICANCE", 0.05)

    log.info("Methodology: aggregation=%s, selection=%s, transforms=%s, knockouts=%s",
             aggregation, selection_method,
             "custom" if transforms else "linear",
             list(knockout_rules.keys()) if knockout_rules else "none")

    # ── Normalize components ─────────────────────────────────────────
    comps = _normalize_components(T, tt_da, pop_da, veg_da, ntl_da, drt_da,
                                  overlays, params=params)

    # Add v2 components (normalize with transforms if specified)
    _add_v2_components(comps, tt_health_da, tt_edu_da, bldg_dens_da,
                       dre_demand_da, transforms, params)

    # ── Apply transforms (v2: fuzzy membership) ──────────────────────
    if transforms:
        log.info("Applying fuzzy transforms...")
        for alias, spec in transforms.items():
            if alias in comps and comps[alias] is not None:
                comps[alias] = fuzzy_transform(comps[alias], spec)
                log.info("  %s → %s (invert=%s)", alias, spec.kind, spec.invert)

    # ── Aggregate ────────────────────────────────────────────────────
    # Build raw values dict for knockout evaluation (before transform)
    knockout_values = {}
    if knockout_rules:
        knockout_values["cropland"] = crop_da
        knockout_values["population"] = pop_da
        knockout_values["gsl"] = veg_da
        knockout_values["drought"] = drt_da

    if aggregation == "geometric":
        log.info("Using geometric aggregation (non-compensatory)...")
        # Build weight dict from all components
        all_weights = _build_weight_dict(comps, params)
        score = geometric_aggregate(
            comps, all_weights,
            knockout_rules=knockout_rules,
            knockout_values=knockout_values,
        )
        weights_norm_for_meta = {k: v / sum(all_weights.values())
                                 for k, v in all_weights.items() if v > 0 and k in comps}
    else:
        # Legacy additive (backward compat)
        score, weights_norm_for_meta = _combine_with_weights(comps, params=params)

    # Apply masks
    score = _apply_masks(score, T, rur_da, crop_da, params=params)

    # Smooth if requested
    score = _smooth_if_needed(score, params=params)

    # Write continuous score
    write_gtiff_masked(score, PRIORITY_TIF, like=T, nodata=np.nan)
    log.info("Wrote %s", Path(PRIORITY_TIF).name)

    if WRITE_JSON_SIDECARS:
        write_geo_sidecar(Path(PRIORITY_TIF), like=T, extra={"kind": "priority_score"})

    # ── Select priority areas ────────────────────────────────────────
    if selection_method == "hotspot":
        log.info("Using Getis-Ord Gi* hotspot detection (p<%.3f)...", hotspot_sig)
        z_scores, mask = getis_ord_gi_star(
            score, bandwidth_cells=5, significance=hotspot_sig)
        # Write z-score surface for inspection
        write_gtiff_masked(z_scores, out_r("priority_gi_star_z"), like=T, nodata=np.nan)
        mask = mask.astype("float32")
    else:
        # Legacy percentile selection
        mask = select_top(
            score, T,
            top_pct=params.TOP_PCT_CELLS if params.TOP_KM2 is None else None,
            top_km2=params.TOP_KM2
        )

    # Optional pruning of tiny blobs
    mask = prune_clusters(mask, int(params.MIN_CLUSTER_CELLS or 0))

    # Policy-aware AOI mask at sink
    mask = apply_aoi_mask_if_enabled(mask, T)

    write_gtiff_masked(mask, PRIORITY_TOP10_TIF, like=T, nodata=np.nan)
    log.info("Wrote %s | selected=%d cells", Path(PRIORITY_TOP10_TIF).name,
             int((mask.values == 1).sum()))

    if WRITE_JSON_SIDECARS:
        write_geo_sidecar(Path(PRIORITY_TOP10_TIF), like=T, extra={"kind": "priority_top_mask"})

    # ── JSON sidecar for reproducibility ─────────────────────────────
    if WRITE_JSON_SIDECARS:
        import json as _json

        meta = {
            "aoi": AOI,
            "preset": ACTIVE_PRESET,
            "methodology": {
                "aggregation": aggregation,
                "selection_method": selection_method,
                "transforms": {k: {"kind": v.kind, "invert": v.invert,
                                   "a": v.a, "b": v.b, "c": v.c, "d": v.d}
                               for k, v in (transforms or {}).items()},
                "knockout_rules": knockout_rules or {},
                "hotspot_significance": hotspot_sig if selection_method == "hotspot" else None,
            },
            "use_components": {
                "ACCESS": int(params.USE_COMPONENTS[0]),
                "POP":    int(params.USE_COMPONENTS[1]),
                "VEG":    int(params.USE_COMPONENTS[2]),
                "NTL":    int(params.USE_COMPONENTS[3]),
                "DRT":    int(params.USE_COMPONENTS[4]),
            },
            "weights_normalized": {k: float(v) for k, v in weights_norm_for_meta.items()},
            "v2_components_present": sorted([
                k for k in ("TT_HEALTH", "TT_EDUCATION", "BUILDING_DENSITY", "DRE_DEMAND")
                if k in comps and comps[k] is not None
            ]),
            "masks": {
                "require_rural": bool(params.MASK_REQUIRE_RURAL),
                "min_cropland": float(params.MASK_MIN_CROPLAND or 0.0),
            },
            "smoothing_radius_cells": int(params.SMOOTH_RADIUS or 0),
            "selection_rule": {
                "method": selection_method,
                "top_pct_cells": params.TOP_PCT_CELLS,
                "top_km2": params.TOP_KM2,
                "min_cluster_cells": int(params.MIN_CLUSTER_CELLS or 0),
            },
            "outputs": {
                "priority_score_tif": Path(PRIORITY_TIF).name,
                "priority_top_mask_tif": Path(PRIORITY_TOP10_TIF).name,
            },
            "overlays_present": sorted([k for k in ("POV","FOOD","MTT","RWI") if k in comps]),
        }
        _meta_path = Path(PRIORITY_TIF).with_suffix(".meta.json")
        _meta_path.write_text(_json.dumps(meta, indent=2))
        log.info("Wrote sidecar meta → %s", _meta_path.name)

    # Admin-2 priority ranking table
    _write_admin2_rank_table(T)

    log.info("Step 07 complete.")


# -------------------------------------------------------------------------
# Admin-2 priority ranking (extracted from main for readability)
# -------------------------------------------------------------------------

def _to_2d_float(arr) -> np.ndarray:
    """Squeeze a DataArray to a 2-D float64 numpy array, handling masked arrays."""
    a = np.asarray(arr.values)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if np.ma.isMaskedArray(a):
        a = a.filled(np.nan)
    if a.ndim != 2:
        raise RuntimeError(f"Expected 2D raster, got shape={a.shape}")
    return a.astype("float64")


def _write_admin2_rank_table(T: xr.DataArray) -> None:
    """
    Build and write the Admin-2 priority ranking table.

    Reads the admin-2 id grid, priority score/mask, and lookup table produced
    by earlier steps, then aggregates per-zone statistics.
    """
    try:
        p_id = Path(ADMIN2_ID_TIF)
        p_lut = Path(ADMIN2_LUT_CSV)

        # Fallback discovery if the exact names differ
        if not p_id.exists():
            cand = list(Path(PATHS.OUT_R).glob(f"{AOI}*admin2*id*1km*.tif"))
            if cand:
                p_id = cand[0]
        if not p_lut.exists():
            cands = (list(Path(PATHS.OUT_T).glob(f"{AOI}*admin2*lookup*.csv")) +
                     list(Path(PATHS.OUT_T).glob(f"{AOI}*admin2*lookup*.csv.gz")))
            if cands:
                p_lut = cands[0]

        if not (p_id.exists() and p_lut.exists()):
            log.info("Admin-2 grid/lookup not found; skip Admin-2 ranking table.")
            return

        idgrid = rxr.open_rasterio(p_id, masked=True).squeeze()
        idgrid = ensure_aligned(idgrid, T, resampling=Resampling.nearest)

        v = _to_2d_float(xr.open_dataarray(PRIORITY_TIF).squeeze())
        m = _to_2d_float(xr.open_dataarray(PRIORITY_TOP10_TIF).squeeze())

        ids = np.asarray(idgrid.values)
        if ids.ndim == 3 and ids.shape[0] == 1:
            ids = ids[0]
        if np.ma.isMaskedArray(ids):
            ids = ids.filled(0)
        ids = np.where(np.isfinite(ids), ids, 0).astype("int32")

        if v.shape != ids.shape or m.shape != ids.shape:
            raise RuntimeError(f"Shape mismatch: score{v.shape}, mask{m.shape}, ids{ids.shape}")

        lut = pd.read_csv(p_lut)
        for col in ["lab", "ADM2CD_c", "NAM_1", "NAM_2"]:
            if col not in lut.columns:
                lut[col] = np.nan
        lut = lut[["lab", "ADM2CD_c", "NAM_1", "NAM_2"]].copy()
        lut["lab"] = lut["lab"].astype("Int64")

        valid = np.isfinite(v) & (ids > 0)
        if not valid.any():
            log.warning("Admin-2 ranking: no valid cells; skipping table.")
            return

        vv  = v[valid]
        ii  = ids[valid]
        sel = np.nan_to_num(m[valid], nan=0.0) > 0

        ak = np.asarray(cell_area_km2_latlon(T).values)
        if ak.ndim == 3 and ak.shape[0] == 1:
            ak = ak[0]
        akv = ak[valid]

        max_id = int(ii.max()) if ii.size else 0
        if max_id == 0:
            log.warning("Admin-2 ranking: all admin ids are zero; skipping table.")
            return

        sums_score   = np.bincount(ii, weights=vv, minlength=max_id + 1)
        cnts_total   = np.bincount(ii, minlength=max_id + 1)
        cnts_sel     = np.bincount(ii, weights=sel.astype("float64"), minlength=max_id + 1)
        km2_total    = np.bincount(ii, weights=akv, minlength=max_id + 1)
        km2_selected = np.bincount(ii, weights=akv * sel.astype("float64"), minlength=max_id + 1)

        means = np.divide(sums_score, cnts_total,
                          out=np.full_like(sums_score, np.nan, dtype="float64"),
                          where=cnts_total > 0)
        share_sel = np.divide(cnts_sel, cnts_total,
                              out=np.zeros_like(cnts_sel, dtype="float64"),
                              where=cnts_total > 0)

        df = pd.DataFrame({
            "lab": np.arange(0, max_id + 1, dtype=int),
            "score": means,
            "selected_cells": cnts_sel.astype(int),
            "total_cells": cnts_total.astype(int),
            "selected_km2": km2_selected.astype("float64"),
            "total_km2": km2_total.astype("float64"),
            "share_selected": share_sel.astype("float64"),
        })
        df = df[df["lab"] > 0]

        out_df = df.merge(lut, on="lab", how="left")
        out_df["selected"]    = out_df["selected_cells"] > 0
        out_df["top10_cells"] = out_df["selected_cells"].astype(int)
        out_df["top10_km2"]   = out_df["selected_km2"].astype("float64")

        out_df = out_df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)
        out_df["rank"] = (np.arange(len(out_df)) + 1).astype(int)
        out_df = _ensure_rank_columns(out_df)

        out_csv = _admin2_rank_path().with_suffix(".csv")
        out_df.to_csv(out_csv, index=False)
        log.info("Wrote Admin-2 priority table → %s (rows=%d, cols=%d)",
                 out_csv.name, len(out_df), out_df.shape[1])

    except Exception as e:
        log.warning(f"Admin-2 priority table skipped due to error: {e}")


if __name__ == "__main__":
    main()

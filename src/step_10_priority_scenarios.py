"""
Step 10 — Priority Sensitivity / Scenario Sweep

Purpose
-------
Quantify how the priority map changes when you toggle components, masks, smoothing,
and Top-X selection. For each scenario we compute:
- selected cells (#) and selected area (km², area-true)
- population covered (sum over selected cells)
- cropland area inside selected (km²)
- average drought in selected (optional, if raster present)
- overlap with a chosen baseline mask (Jaccard index & overlap %)

Design choices
--------------
- Scenarios are defined in a local SCENARIOS list here for simplicity.
  You can move them to config or a CSV later.
- We *temporarily* override PARAMS for each scenario, then restore the original.
- If a municipal poverty/food raster exists (Step 06) it’s INCLUDED via Step-07 logic
  (we call a local replica of Step-07 normalization to avoid cross-module coupling).
- Output includes per-scenario CSV summary; optional per-scenario masks can be saved.

Inputs (expected)
-----------------
- PARAMS.TARGET_GRID (1-km travel time, minutes)
- outputs/rasters/{AOI}_pop_1km.tif
- outputs/rasters/{AOI}_cropland_fraction_1km.tif
- outputs/rasters/{AOI}_rural_1km.tif
- outputs/rasters/{AOI}_drought_1km.tif (optional)
- outputs/rasters/{AOI}_veg_1km.tif (optional)
- outputs/rasters/{AOI}_ntl_1km.tif (optional)
- outputs/rasters/{AOI}_priority_top10_mask.tif (baseline, optional; otherwise baseline is scenario[0])

Outputs
-------
- outputs/tables/{AOI}_priority_scenarios_summary.csv
- (optional) outputs/rasters/{AOI}_priority_mask_{scenario_id}.tif  (if SAVE_MASKS=True)

Usage
-----
- Edit SCENARIOS below (or keep defaults) and run:
    import importlib
    m = importlib.import_module("step_10_priority_scenarios")
    importlib.reload(m); m.main()

"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from scipy.ndimage import label

from config import (
    AOI, PATHS, PARAMS, Params, TransformSpec, get_logger, out_r, out_t,
    PRIORITY_TOP10_TIF,
    RESAMPLE_DEFAULT_CONT, RESAMPLE_DEFAULT_CAT,
    WRITE_JSON_SIDECARS, write_geo_sidecar,
    MUNI_TT_FIELD_MINUTES,
    PRESETS, ACTIVE_PRESET, preset_to_params, ThematicPreset,
)

from utils_geo import (
    open_template, write_gtiff_masked,
    focal_mean, cell_area_km2_latlon,
    ensure_aligned, open_and_align,
    # v2 analytical functions
    fuzzy_transform,
    geometric_aggregate,
    getis_ord_gi_star,
)

log = get_logger(__name__)

# ==========================
# Scenario configuration
# ==========================

@dataclass
class Scenario:
    id: str
    desc: str
    USE_COMPONENTS: Tuple[int,int,int,int,int]  # ACC, POP, VEG, NTL, DRT
    W_ACC: float
    W_POP: float
    W_VEG: float
    W_NTL: float
    W_DRT: float
    MASK_REQUIRE_RURAL: bool
    MASK_MIN_CROPLAND: float  # 0 disables
    SMOOTH_RADIUS: int        # 0,1,2
    TOP_PCT_CELLS: Optional[float] = 0.10    # use pct OR km2
    TOP_KM2: Optional[float] = None          # set to number to use km² selection
    MIN_CLUSTER_CELLS: int = 0               # 0 disables
    # Overlay weights (default = inherit from global PARAMS)
    W_POV: float = PARAMS.W_POV
    W_FOOD: float = PARAMS.W_FOOD
    W_MTT: float = PARAMS.W_MTT
    W_RWI: float = PARAMS.W_RWI
    # Transform caps
    NTL_CAP: float = PARAMS.NTL_CAP
    VEG_MIN: float = PARAMS.VEG_MIN
    # Link to source preset (None = manually defined scenario)
    preset_name: Optional[str] = None

    # --- v2 component weights ---
    W_TT_HEALTH: float = 0.0
    W_TT_EDUCATION: float = 0.0
    W_BUILDING_DENSITY: float = 0.0
    W_DRE_DEMAND: float = 0.0

    # --- v2 methodology ---
    AGGREGATION: str = "additive"             # "additive" | "geometric"
    SELECTION_METHOD: str = "percentile"      # "percentile" | "hotspot"
    TRANSFORMS: Optional[Dict[str, TransformSpec]] = None
    KNOCKOUT_RULES: Optional[Dict[str, float]] = None
    HOTSPOT_SIGNIFICANCE: float = 0.05

# Sensible defaults to start—tune freely
SCENARIOS: List[Scenario] = [
    Scenario(
        id="baseline",
        desc="ACC+POP+DRT, rural-only, min crop 5%, 3x3 smooth, Top10%",
        USE_COMPONENTS=(1,1,0,0,1),
        W_ACC=0.50, W_POP=0.30, W_VEG=0.00, W_NTL=0.00, W_DRT=0.20,
        MASK_REQUIRE_RURAL=True,
        MASK_MIN_CROPLAND=0.05,
        SMOOTH_RADIUS=1,
        TOP_PCT_CELLS=0.10, TOP_KM2=None,
        MIN_CLUSTER_CELLS=30
    ),
    Scenario(
        id="no_drought",
        desc="ACC+POP only; rural-only; min crop 5%; Top10%",
        USE_COMPONENTS=(1,1,0,0,0),
        W_ACC=0.70, W_POP=0.30, W_VEG=0.00, W_NTL=0.00, W_DRT=0.00,
        MASK_REQUIRE_RURAL=True,
        MASK_MIN_CROPLAND=0.05,
        SMOOTH_RADIUS=1,
        TOP_PCT_CELLS=0.10, TOP_KM2=None,
        MIN_CLUSTER_CELLS=30
    ),
    Scenario(
        id="veg_signal",
        desc="ACC+POP+VEG; rural-only; min crop 10%; Top10%",
        USE_COMPONENTS=(1,1,1,0,0),
        W_ACC=0.50, W_POP=0.30, W_VEG=0.20, W_NTL=0.00, W_DRT=0.00,
        MASK_REQUIRE_RURAL=True,
        MASK_MIN_CROPLAND=0.10,
        SMOOTH_RADIUS=1,
        TOP_PCT_CELLS=0.10, TOP_KM2=None,
        MIN_CLUSTER_CELLS=30
    ),
    Scenario(
        id="top_800km2",
        desc="ACC+POP+DRT, rural-only; select fixed 800 km²; mild smooth",
        USE_COMPONENTS=(1,1,0,0,1),
        W_ACC=0.50, W_POP=0.30, W_VEG=0.00, W_NTL=0.00, W_DRT=0.20,
        MASK_REQUIRE_RURAL=True,
        MASK_MIN_CROPLAND=0.05,
        SMOOTH_RADIUS=1,
        TOP_PCT_CELLS=None, TOP_KM2=800.0,
        MIN_CLUSTER_CELLS=30
    ),
]

def scenarios_from_presets(
    presets: dict[str, ThematicPreset] | None = None,
) -> list[Scenario]:
    """Generate one ``Scenario`` per ``ThematicPreset``.

    All v2 methodology fields (aggregation, selection, transforms,
    knockouts) are propagated so each scenario faithfully reproduces
    its preset's full analytical pipeline.
    """
    if presets is None:
        presets = PRESETS
    return [
        Scenario(
            id=p.name,
            desc=p.description,
            USE_COMPONENTS=p.USE_COMPONENTS,
            W_ACC=p.W_ACC, W_POP=p.W_POP, W_VEG=p.W_VEG,
            W_NTL=p.W_NTL, W_DRT=p.W_DRT,
            W_POV=p.W_POV, W_FOOD=p.W_FOOD, W_MTT=p.W_MTT, W_RWI=p.W_RWI,
            NTL_CAP=p.NTL_CAP, VEG_MIN=p.VEG_MIN,
            MASK_REQUIRE_RURAL=p.MASK_REQUIRE_RURAL,
            MASK_MIN_CROPLAND=p.MASK_MIN_CROPLAND,
            SMOOTH_RADIUS=p.SMOOTH_RADIUS,
            TOP_PCT_CELLS=p.TOP_PCT_CELLS, TOP_KM2=p.TOP_KM2,
            MIN_CLUSTER_CELLS=p.MIN_CLUSTER_CELLS,
            preset_name=p.name,
            # v2 component weights
            W_TT_HEALTH=p.W_TT_HEALTH,
            W_TT_EDUCATION=p.W_TT_EDUCATION,
            W_BUILDING_DENSITY=p.W_BUILDING_DENSITY,
            W_DRE_DEMAND=p.W_DRE_DEMAND,
            # v2 methodology
            AGGREGATION=p.AGGREGATION,
            SELECTION_METHOD=p.SELECTION_METHOD,
            TRANSFORMS=p.TRANSFORMS,
            KNOCKOUT_RULES=p.KNOCKOUT_RULES,
            HOTSPOT_SIGNIFICANCE=p.HOTSPOT_SIGNIFICANCE,
        )
        for p in presets.values()
    ]


# Toggle: use preset-derived scenarios (True) or legacy hardcoded (False)
USE_PRESET_SCENARIOS = True

if USE_PRESET_SCENARIOS:
    SCENARIOS = scenarios_from_presets()

# Save per-scenario masks?
SAVE_MASKS = True


# ==========================
# Helpers (local copy of Step-07 logic, simplified)
# ==========================

def _r(path: Path) -> xr.DataArray | None:
    if not Path(path).exists():
        return None
    return rxr.open_rasterio(path, masked=True).squeeze()

def _find_optional_muni_rasters(T: xr.DataArray) -> Dict[str, xr.DataArray]:
    """Optional municipal overlays from Step-06."""
    cand = {
        "poverty": out_r("muni_poverty_poverty_rural_1km"),
        "food": out_r("muni_foodinsecurity_food_insec_scale_1km"),
        "muni_tt": out_r(f"muni_traveltime_{MUNI_TT_FIELD_MINUTES}_1km"),
    }
    out = {}
    for k, p in cand.items():
        da = open_and_align(p, T, resampling=RESAMPLE_DEFAULT_CONT)
        if da is not None:
            out[k] = da
    return out

def _safe_minmax_scale(da: xr.DataArray, lo: float, hi: float, invert: bool = False) -> xr.DataArray:
    out = da.clip(lo, hi)
    out = (out - lo) / (hi - lo + 1e-9)
    return (1.0 - out) if invert else out

def _normalize_components(T, tt_da, pop_da, veg_da, ntl_da, drt_da, muni,
                          v2_layers, transforms, params=PARAMS):
    """
    Normalize the component rasters to 0..1 and return a dict of DataArrays.
    Uses per-scenario 'params' (immutable) instead of mutating global PARAMS.

    When ``transforms`` specifies a custom TransformSpec for a component,
    the raw (un-normalised) value is stored — fuzzy_transform is applied
    later in the pipeline.

    Parameters
    ----------
    v2_layers : dict
        Keys: tt_health, tt_edu, bldg_dens, dre_demand (DataArray or None).
    transforms : dict[str, TransformSpec] | None
        Per-indicator fuzzy membership specs from the scenario/preset.
    """
    comps = {}

    # Access: invert minutes; clamp using max ISO threshold
    max_iso = float(max(params.ISO_THRESH))
    if transforms and "ACC" in transforms:
        comps["ACC"] = tt_da  # raw — transform later
    else:
        comps["ACC"] = _safe_minmax_scale(tt_da, 0.0, max_iso, invert=True)

    # Population: robust cap at P95 to avoid single-cell spikes
    if transforms and "POP" in transforms:
        comps["POP"] = pop_da  # raw
    else:
        v = pop_da.values.astype("float32")
        v[v < 0] = np.nan
        p95 = np.nanpercentile(v, 95.0)
        p95 = p95 if np.isfinite(p95) and p95 > 0 else (np.nanmax(v) or 1.0)
        comps["POP"] = _safe_minmax_scale(pop_da, 0.0, float(p95))

    # Vegetation (optional)
    if veg_da is not None:
        if transforms and "VEG" in transforms:
            comps["VEG"] = veg_da
        else:
            veg_min = float(params.VEG_MIN)
            comps["VEG"] = _safe_minmax_scale(veg_da.clip(veg_min, 1.0), veg_min, 1.0)
    else:
        comps["VEG"] = None

    # Night lights (optional)
    if ntl_da is not None:
        if transforms and "NTL" in transforms:
            comps["NTL"] = ntl_da
        else:
            ntl_cap = float(params.NTL_CAP)
            comps["NTL"] = _safe_minmax_scale(ntl_da.clip(0.0, ntl_cap), 0.0, ntl_cap)
    else:
        comps["NTL"] = None

    # Drought (optional) — many datasets are 0..1 already; handle 0..100, invert
    if drt_da is not None:
        if transforms and "DRT" in transforms:
            comps["DRT"] = drt_da
        else:
            med = float(np.nanmedian(drt_da.values))
            dr = drt_da/100.0 if med > 1.0 else drt_da
            drc = dr.clip(0.0, 0.30)
            comps["DRT"] = _safe_minmax_scale(drc, 0.0, 0.30, invert=True)
    else:
        comps["DRT"] = None

    # Admin2 overlays already normalized by Step 06, but keep keys consistent
    if "poverty" in muni: comps["POV"]  = muni["poverty"].clip(0.0, 1.0)
    if "food" in muni:
        if transforms and "FOOD" in transforms:
            comps["FOOD"] = muni["food"]
        else:
            comps["FOOD"] = muni["food"].clip(0.0, 1.0)
    if "muni_tt" in muni: comps["MTT"]  = _safe_minmax_scale(muni["muni_tt"], 0.0, max_iso, invert=True)

    # --- v2 layers ---
    _add_v2_components(comps, v2_layers, transforms, params)

    return comps


def _add_v2_components(comps, v2_layers, transforms, params):
    """Add v2 component layers (health/education access, buildings, DRE)."""
    max_tt = float(max(params.ISO_THRESH))

    # Travel time to health
    tt_health = v2_layers.get("tt_health")
    if tt_health is not None:
        if transforms and "TT_HEALTH" in transforms:
            comps["TT_HEALTH"] = tt_health
        else:
            comps["TT_HEALTH"] = _safe_minmax_scale(tt_health, 0.0, max_tt, invert=True)

    # Travel time to education
    tt_edu = v2_layers.get("tt_edu")
    if tt_edu is not None:
        if transforms and "TT_EDUCATION" in transforms:
            comps["TT_EDUCATION"] = tt_edu
        else:
            comps["TT_EDUCATION"] = _safe_minmax_scale(tt_edu, 0.0, max_tt, invert=True)

    # Building density
    bldg = v2_layers.get("bldg_dens")
    if bldg is not None:
        if transforms and "BUILDING_DENSITY" in transforms:
            comps["BUILDING_DENSITY"] = bldg
        else:
            v = bldg.values.astype("float32")
            v[v < 0] = np.nan
            p95 = np.nanpercentile(v, 95.0) if np.any(np.isfinite(v)) else 1.0
            p95 = p95 if np.isfinite(p95) and p95 > 0 else 1.0
            comps["BUILDING_DENSITY"] = _safe_minmax_scale(bldg, 0.0, float(p95))

    # DRE demand
    dre = v2_layers.get("dre_demand")
    if dre is not None:
        if transforms and "DRE_DEMAND" in transforms:
            comps["DRE_DEMAND"] = dre
        else:
            v = dre.values.astype("float32")
            v[v < 0] = np.nan
            p95 = np.nanpercentile(v, 95.0) if np.any(np.isfinite(v)) else 1.0
            p95 = p95 if np.isfinite(p95) and p95 > 0 else 1.0
            comps["DRE_DEMAND"] = _safe_minmax_scale(dre, 0.0, float(p95))


def _build_weight_dict(comps, params=PARAMS):
    """Build complete weight dictionary for all available components."""
    weights = {}
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

    # Overlay weights
    for key, attr in [("POV", "W_POV"), ("FOOD", "W_FOOD"),
                      ("MTT", "W_MTT"), ("RWI", "W_RWI")]:
        if key in comps and comps[key] is not None:
            w = float(getattr(params, attr, 0.0))
            if w > 0:
                weights[key] = w

    # v2 component weights
    for key, attr in [("TT_HEALTH", "W_TT_HEALTH"),
                      ("TT_EDUCATION", "W_TT_EDUCATION"),
                      ("BUILDING_DENSITY", "W_BUILDING_DENSITY"),
                      ("DRE_DEMAND", "W_DRE_DEMAND")]:
        if key in comps and comps[key] is not None:
            w = float(getattr(params, attr, 0.0))
            if w > 0:
                weights[key] = w

    return {k: v for k, v in weights.items() if v > 0}


def _combine_with_weights(comps, params=PARAMS):
    """
    Linear blend of enabled components with weights that sum to 1.
    Honors per-scenario toggles in 'params.USE_COMPONENTS'.
    Includes overlay weights (POV, FOOD, MTT, RWI) and v2 weights when present.
    """
    weights = _build_weight_dict(comps, params)

    if not weights:
        raise RuntimeError("No components enabled for this scenario.")

    w_sum = sum(weights.values())
    W_norm = {k: v / w_sum for k, v in weights.items()}

    out = None
    for key, w in W_norm.items():
        da = comps[key]
        out = da * w if out is None else out + da * w

    return out.clip(0.0, 1.0)


def _apply_masks(score, rural_da, cropfrac_da, params=PARAMS):
    """
    Apply rural-only mask and minimum cropland fraction, using per-scenario params.
    """
    out = score.copy()

    # Rural-only
    if bool(params.MASK_REQUIRE_RURAL):
        if rural_da is None:
            log.warning("MASK_REQUIRE_RURAL=True but no rural raster; skipping rural mask.")
        else:
            out = out.where(rural_da > 0.5)

    # Min cropland fraction
    min_cf = float(params.MASK_MIN_CROPLAND or 0.0)
    if min_cf > 0.0:
        if cropfrac_da is None:
            log.warning("MASK_MIN_CROPLAND>0 but no cropland raster; skipping crop mask.")
        else:
            out = out.where(cropfrac_da >= min_cf)

    return out


def _select_top_mask(score, T, params=PARAMS):
    """
    Convert continuous score to a binary mask using either TOP_KM2 (area-based)
    or TOP_PCT_CELLS (percentile).
    """
    v = score.values
    valid = np.isfinite(v)
    if not np.any(valid):
        return score * 0

    # Area-based selection
    if params.TOP_KM2 is not None:
        area = cell_area_km2_latlon(T).values
        area[~valid] = 0.0

        flat_idx = np.argsort(v[valid])[::-1]        # high→low
        v_valid  = v[valid][flat_idx]
        a_valid  = area[valid][flat_idx]
        cum_area = np.cumsum(a_valid)

        cutoff = float(params.TOP_KM2)
        k = int(np.searchsorted(cum_area, cutoff, side="left")) + 1

        sel = np.zeros_like(v, dtype=np.uint8)
        vi = np.where(valid)
        sel[vi[0][flat_idx[:k]], vi[1][flat_idx[:k]]] = 1
        return xr.DataArray(sel, coords=score.coords, dims=score.dims)

    # Percentile-based selection
    top_pct = float(params.TOP_PCT_CELLS or 0.10)
    q = np.nanpercentile(v, (1.0 - top_pct) * 100.0)
    return (score >= q).astype(np.uint8)


def _remove_small_clusters(mask, params=PARAMS):
    """
    Drop connected components smaller than MIN_CLUSTER_CELLS.
    """
    min_cells = int(params.MIN_CLUSTER_CELLS or 0)
    if min_cells <= 1:
        return mask

    arr = (mask.values > 0).astype(np.uint8)
    lbl, n = label(arr)
    if n == 0:
        return mask

    sizes = np.bincount(lbl.ravel())
    kill = np.where(sizes < min_cells)[0]
    kill = kill[kill != 0]  # ignore background

    if len(kill) == 0:
        return mask

    out = arr.copy()
    for lab in kill:
        out[lbl == lab] = 0

    return xr.DataArray(out.astype(np.uint8), coords=mask.coords, dims=mask.dims)

# ==========================
# Scenario runner
# ==========================

def _compute_priority_mask_for_current_params(T, rasters, params=PARAMS, scn=None):
    """
    Full pipeline for a scenario: normalize → [fuzzy transform] →
    aggregate → masks → smooth → select → clean.

    Supports both legacy additive + percentile AND v2 geometric + hotspot.
    Everything keyed off the provided 'params' (immutable) and scenario
    methodology fields.

    Parameters
    ----------
    scn : Scenario | None
        When provided, v2 methodology is read from the scenario dataclass.
        When None, falls back to legacy additive + percentile.
    """
    # Inputs
    tt     = T                # travel time grid (minutes) used as ACCESS
    pop    = rasters["pop"]
    veg    = rasters["veg"]
    ntl    = rasters["ntl"]
    drt    = rasters["drt"]
    crop   = rasters["crop"]
    rural  = rasters["rural"]
    muni   = rasters["muni"]
    v2_layers = rasters.get("v2", {})

    # Methodology from scenario (or defaults)
    aggregation = getattr(scn, "AGGREGATION", "additive") if scn else "additive"
    selection_method = getattr(scn, "SELECTION_METHOD", "percentile") if scn else "percentile"
    transforms = getattr(scn, "TRANSFORMS", None) if scn else None
    knockout_rules = getattr(scn, "KNOCKOUT_RULES", None) if scn else None
    hotspot_sig = getattr(scn, "HOTSPOT_SIGNIFICANCE", 0.05) if scn else 0.05

    # 1) Normalize components (with transform-awareness)
    comps = _normalize_components(T, tt, pop, veg, ntl, drt, muni,
                                  v2_layers, transforms, params=params)

    # 2) Apply fuzzy transforms where specified
    if transforms:
        for alias, spec in transforms.items():
            if alias in comps and comps[alias] is not None:
                comps[alias] = fuzzy_transform(comps[alias], spec)

    # 3) Aggregate
    if aggregation == "geometric":
        all_weights = _build_weight_dict(comps, params)
        # Build knockout values from raw layers
        knockout_values = {}
        if knockout_rules:
            knockout_values["cropland"] = crop
            knockout_values["population"] = pop
            knockout_values["gsl"] = veg
            knockout_values["drought"] = drt
        score = geometric_aggregate(
            comps, all_weights,
            knockout_rules=knockout_rules,
            knockout_values=knockout_values,
        )
    else:
        # Legacy additive
        score = _combine_with_weights(comps, params=params)

    # 4) Apply masks
    score = _apply_masks(score, rural, crop, params=params)

    # 5) Optional smoothing
    r = int(params.SMOOTH_RADIUS or 0)
    if r > 0:
        score = focal_mean(score, radius=r)

    # 6) Select priority areas
    if selection_method == "hotspot":
        _, mask = getis_ord_gi_star(score, bandwidth_cells=5,
                                    significance=hotspot_sig)
        mask = mask.astype(np.uint8)
    else:
        mask = _select_top_mask(score, T, params=params)

    mask = _remove_small_clusters(mask, params=params)

    return mask


def _load_all_inputs(T) -> Dict[str, xr.DataArray]:
    """
    Load & align all rasters we may need, including v2 layers.
    """
    out = {
        "pop":   open_and_align(PATHS.OUT_R / f"{AOI}_pop_1km.tif", T, RESAMPLE_DEFAULT_CONT),
        "veg":   open_and_align(PATHS.OUT_R / f"{AOI}_veg_1km.tif", T, RESAMPLE_DEFAULT_CONT),
        "ntl":   open_and_align(PATHS.OUT_R / f"{AOI}_ntl_1km.tif", T, RESAMPLE_DEFAULT_CONT),
        "drt":   open_and_align(PATHS.OUT_R / f"{AOI}_drought_1km.tif", T, RESAMPLE_DEFAULT_CONT),
        "crop":  open_and_align(PATHS.OUT_R / f"{AOI}_cropland_fraction_1km.tif", T, RESAMPLE_DEFAULT_CONT),
        "rural": open_and_align(PATHS.OUT_R / f"{AOI}_rural_1km.tif", T, RESAMPLE_DEFAULT_CAT),
    }
    out["muni"] = _find_optional_muni_rasters(T)

    # v2 layers (optional — Step 00 may not have produced them yet)
    out["v2"] = {
        "tt_health":  open_and_align(out_r("tt_health_motorised_1km"), T, RESAMPLE_DEFAULT_CONT),
        "tt_edu":     open_and_align(out_r("tt_education_motorised_1km"), T, RESAMPLE_DEFAULT_CONT),
        "bldg_dens":  open_and_align(out_r("building_density_1km"), T, RESAMPLE_DEFAULT_CONT),
        "dre_demand": open_and_align(out_r("dre_demand_density_1km"), T, RESAMPLE_DEFAULT_CONT),
    }
    v2_present = [k for k, v in out["v2"].items() if v is not None]
    if v2_present:
        log.info("v2 layers loaded: %s", ", ".join(v2_present))

    if out["pop"] is None:
        raise RuntimeError("Missing required raster: pop_1km")
    return out

def _scenario_metrics(mask: xr.DataArray, T: xr.DataArray, rasters: Dict[str,xr.DataArray]) -> Dict[str, float]:
    """
    Compute key metrics for a selected mask, including v2 indicators.
    """
    area = cell_area_km2_latlon(T)
    m = (mask.values > 0)
    km2 = float(np.nansum(area.values[m]))
    n_cells = int(m.sum())
    pop = rasters["pop"].values
    pop_sum = float(np.nansum(pop[m]))
    crop = rasters["crop"]; crop_km2 = float(np.nansum((crop.values * area.values)[m])) if crop is not None else np.nan
    drt = rasters["drt"]
    drt_mean = float(np.nanmean(drt.values[m])) if drt is not None and n_cells>0 else np.nan

    metrics = {
        "selected_cells": n_cells,
        "selected_km2": km2,
        "pop_selected": pop_sum,
        "ag_km2_selected": crop_km2,
        "drought_mean_selected": drt_mean,
    }

    # v2 metrics (if layers present)
    v2 = rasters.get("v2", {})
    if v2.get("tt_health") is not None and n_cells > 0:
        metrics["tt_health_mean_selected"] = float(np.nanmean(v2["tt_health"].values[m]))
    if v2.get("tt_edu") is not None and n_cells > 0:
        metrics["tt_edu_mean_selected"] = float(np.nanmean(v2["tt_edu"].values[m]))
    if v2.get("bldg_dens") is not None and n_cells > 0:
        metrics["bldg_dens_mean_selected"] = float(np.nanmean(v2["bldg_dens"].values[m]))
    if v2.get("dre_demand") is not None and n_cells > 0:
        metrics["dre_demand_sum_selected"] = float(np.nansum(v2["dre_demand"].values[m]))

    return metrics

def _overlap_stats(a: xr.DataArray, b: xr.DataArray) -> Dict[str, float]:
    """
    Compute overlap% (A∩B / A), and Jaccard (A∩B / A∪B).
    """
    A = (a.values > 0); B = (b.values > 0)
    inter = float((A & B).sum())
    a_sum = float(A.sum()); b_sum = float(B.sum())
    union = float((A | B).sum())
    return {
        "overlap_pct_vs_baseline": (inter / a_sum) if a_sum > 0 else np.nan,
        "jaccard_vs_baseline": (inter / union) if union > 0 else np.nan,
        "baseline_cells": b_sum,
    }

def _params_for_scenario(scn):
    """
    Create a new Params object with scenario overrides applied,
    without mutating the global (frozen) PARAMS.
    """
    return replace(
        PARAMS,
        USE_COMPONENTS=scn.USE_COMPONENTS,
        W_ACC=scn.W_ACC, W_POP=scn.W_POP, W_VEG=scn.W_VEG, W_NTL=scn.W_NTL, W_DRT=scn.W_DRT,
        W_POV=scn.W_POV, W_FOOD=scn.W_FOOD, W_MTT=scn.W_MTT, W_RWI=scn.W_RWI,
        NTL_CAP=scn.NTL_CAP, VEG_MIN=scn.VEG_MIN,
        MASK_REQUIRE_RURAL=scn.MASK_REQUIRE_RURAL,
        MASK_MIN_CROPLAND=scn.MASK_MIN_CROPLAND,
        SMOOTH_RADIUS=scn.SMOOTH_RADIUS,
        MIN_CLUSTER_CELLS=scn.MIN_CLUSTER_CELLS,
        TOP_PCT_CELLS=scn.TOP_PCT_CELLS,
        TOP_KM2=scn.TOP_KM2,
        # v2 component weights
        W_TT_HEALTH=scn.W_TT_HEALTH,
        W_TT_EDUCATION=scn.W_TT_EDUCATION,
        W_BUILDING_DENSITY=scn.W_BUILDING_DENSITY,
        W_DRE_DEMAND=scn.W_DRE_DEMAND,
    )

def _restore_params(saved_params):
    for k, v in saved_params.items():
        setattr(PARAMS, k, v)

# ==========================
# Main
# ==========================

def main() -> None:
    """
    Run a set of priority scenarios, compute key metrics and overlaps vs baseline,
    and save a tidy summary CSV (and optional masks).
    """
    # Template grid
    T = open_template(PARAMS.TARGET_GRID)
    tf = T.rio.transform(); resx, resy = abs(tf.a), abs(tf.e)
    log.info(f"Target grid | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}")

    # Use a local sidecar flag to avoid scope issues
    _write_sidecar = bool(WRITE_JSON_SIDECARS)

    # Baseline path (canonical Step-07 top-mask)
    baseline_path = Path(PRIORITY_TOP10_TIF)

    # Collect per-scenario metadata for a JSON sidecar
    # Use a dict so we can add top-level fields and a list of scenarios
    scenarios_meta = {
        "aoi": AOI,
        "target_grid": Path(PARAMS.TARGET_GRID).name if isinstance(PARAMS.TARGET_GRID, str) else "in-memory",
        "baseline_source": ("file" if baseline_path.exists() else "first_scenario"),
        "scenarios": []
    }

    # Load rasters needed
    rasters = _load_all_inputs(T)
    
    # Load (or compute) baseline mask
    baseline_path = Path(PRIORITY_TOP10_TIF)
    if baseline_path.exists():
        base_mask = open_and_align(baseline_path, T, RESAMPLE_DEFAULT_CAT)
        log.info(f"Using on-disk baseline: {baseline_path.name}")
    else:
        base_mask = None
        log.info("No on-disk baseline found; will treat the first scenario as baseline.")

    # Remember original PARAMS to restore later
    saved = {
        "USE_COMPONENTS": tuple(PARAMS.USE_COMPONENTS),
        "W_ACC": PARAMS.W_ACC, "W_POP": PARAMS.W_POP, "W_VEG": PARAMS.W_VEG, "W_NTL": PARAMS.W_NTL, "W_DRT": PARAMS.W_DRT,
        "MASK_REQUIRE_RURAL": bool(PARAMS.MASK_REQUIRE_RURAL),
        "MASK_MIN_CROPLAND": float(PARAMS.MASK_MIN_CROPLAND or 0.0),
        "SMOOTH_RADIUS": int(PARAMS.SMOOTH_RADIUS or 0),
        "TOP_PCT_CELLS": PARAMS.TOP_PCT_CELLS,
        "TOP_KM2": PARAMS.TOP_KM2,
        "MIN_CLUSTER_CELLS": int(PARAMS.MIN_CLUSTER_CELLS or 0),
    }

    # -------------------------------------------------------------------------
    # Scenarios: evaluate without mutating global PARAMS (frozen dataclass)
    # -------------------------------------------------------------------------
    rows = []
    written = 0
    base_for_overlap = None

    for idx, scn in enumerate(SCENARIOS):
        log.info(f"Scenario [{scn.id}] — {scn.desc}")

        # Build an immutable per-scenario params object from baseline PARAMS + overrides
        params = _params_for_scenario(scn)

        # Compute mask using only this params instance + v2 methodology from scenario
        mask = _compute_priority_mask_for_current_params(T, rasters, params=params, scn=scn)

        # Persist mask if requested
        if SAVE_MASKS:
            out_mask = PATHS.OUT_R / f"{AOI}_priority_mask_{scn.id}.tif"
            write_gtiff_masked(mask, out_mask, like=T, nodata=np.nan)
            written += 1
            if _write_sidecar:
                write_geo_sidecar(out_mask, like=T, extra={"kind": "scenario_mask", "scenario_id": scn.id})

        # Collect machine-readable scenario meta (weights, masks, selection, outputs)
        scenarios_meta["scenarios"].append({
            "id": scn.id,
            "desc": scn.desc,
            "preset": getattr(scn, "preset_name", None),
            "use_components": list(scn.USE_COMPONENTS),
            "weights": {
                "ACC": float(scn.W_ACC), "POP": float(scn.W_POP),
                "VEG": float(scn.W_VEG), "NTL": float(scn.W_NTL), "DRT": float(scn.W_DRT),
            },
            "overlay_weights": {
                "POV": float(scn.W_POV), "FOOD": float(scn.W_FOOD),
                "MTT": float(scn.W_MTT), "RWI": float(scn.W_RWI),
            },
            "v2_weights": {
                "TT_HEALTH": float(scn.W_TT_HEALTH),
                "TT_EDUCATION": float(scn.W_TT_EDUCATION),
                "BUILDING_DENSITY": float(scn.W_BUILDING_DENSITY),
                "DRE_DEMAND": float(scn.W_DRE_DEMAND),
            },
            "methodology": {
                "aggregation": scn.AGGREGATION,
                "selection_method": scn.SELECTION_METHOD,
                "transforms": {k: {"kind": v.kind, "invert": v.invert,
                                   "a": v.a, "b": v.b, "c": v.c, "d": v.d}
                               for k, v in (scn.TRANSFORMS or {}).items()},
                "knockout_rules": scn.KNOCKOUT_RULES or {},
                "hotspot_significance": (scn.HOTSPOT_SIGNIFICANCE
                                         if scn.SELECTION_METHOD == "hotspot" else None),
            },
            "masks": {
                "require_rural": bool(scn.MASK_REQUIRE_RURAL),
                "min_cropland": float(scn.MASK_MIN_CROPLAND or 0.0),
            },
            "smooth_radius_cells": int(scn.SMOOTH_RADIUS or 0),
            "selection": {
                "top_pct_cells": scn.TOP_PCT_CELLS,
                "top_km2": scn.TOP_KM2,
                "min_cluster_cells": int(scn.MIN_CLUSTER_CELLS or 0),
            },
            "outputs": {
                "mask_tif": (out_mask.name if SAVE_MASKS else None)
            }
        })


        # Metrics for scenario
        m = _scenario_metrics(mask, T, rasters)

        # Overlap against baseline: if an on-disk baseline exists, we already loaded it
        # otherwise, use the first scenario as "baseline in-memory"
        if baseline_path.exists():
            base_for_overlap = base_mask
        else:
            if idx == 0:
                base_for_overlap = mask

        ov = _overlap_stats(mask, base_for_overlap)

        # Collect row
        row = {"scenario_id": scn.id, "desc": scn.desc, **asdict(scn), **m, **ov}
        rows.append(row)

    # Write scenario summary (no need to restore PARAMS anymore)
    df = pd.DataFrame(rows)

    # tidy numerics for readability and stable diffs
    for c in ("selected_km2", "pop_selected", "ag_km2_selected", "drought_mean_selected",
              "overlap_pct_vs_baseline", "jaccard_vs_baseline",
              "tt_health_mean_selected", "tt_edu_mean_selected",
              "bldg_dens_mean_selected", "dre_demand_sum_selected"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64").round(4)

    if "selected_cells" in df.columns:
        df["selected_cells"] = pd.to_numeric(df["selected_cells"], errors="coerce").round(0).astype("Int64")
    if "baseline_cells" in df.columns:
        df["baseline_cells"] = pd.to_numeric(df["baseline_cells"], errors="coerce").round(0).astype("Int64")

    out_csv = out_t("priority_scenarios_summary")
    df.to_csv(out_csv, index=False)

    log.info(f"Wrote scenario summary → {Path(out_csv).name} | scenarios={len(SCENARIOS)} | masks_saved={written}")

    if _write_sidecar:
        import json as _json
        meta_path = PATHS.OUT_T / f"{AOI}_priority_scenarios.meta.json"
        meta_path.write_text(_json.dumps(scenarios_meta, indent=2))
        log.info(f"Wrote scenarios sidecar → {meta_path.name}")

    log.info("Step 10 complete.")


if __name__ == "__main__":
    main()

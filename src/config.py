"""
Lobito Corridor Spatial Analysis — Central Configuration
=======================================================

Scope
-----
This module centralizes:
- Project & AOI awareness (paths, filenames, helpers)
- Logging (timestamped INFO logger)
- Resampling policy (string → rasterio.enums.Resampling)
- Admin2/RAPP ingestion controls and theme wiring
- Optional overlays (e.g., RWI grid)
- Priority surface knobs (components, masks, smoothing, selection)
- Roads filtering policy (ALL or allow-list of OSM fclass)
- Canonical output name helpers (rasters/tables/figs)

Usage (from any step):
    from config import AOI, PATHS, PARAMS
    from config import out_r, out_t, out_f, get_logger, RESAMPLE

Design notes
------------
- Keep AOI in filenames for reproducibility.
- All outputs go under <ROOT>/outputs/{rasters|tables|figs}.
- Steps should **not** assume optional layers exist; handle missing gracefully.
- PARAMS is a frozen dataclass (immutable) to avoid accidental mutation across steps.
"""

from __future__ import annotations

# stdlib
from dataclasses import dataclass
from pathlib import Path
import os
import sys
import logging
from datetime import datetime, timezone
import json

# third-party
from rasterio.enums import Resampling


# ======================================================================
# 1) Logging
# ======================================================================

def get_logger(name: str = "lobito") -> logging.Logger:
    """
    Return a timestamped, INFO-level logger that writes to stdout.

    Example:
        from config import get_logger
        log = get_logger(__name__)
        log.info("message")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler(stream=sys.stdout)
        h.setFormatter(logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
        logger.propagate = False
    return logger


# ======================================================================
# 2) Project root & AOI tag
# ======================================================================

def _detect_project_root() -> Path:
    """
    Detect project root assuming this file lives in `<root>/src/config.py`.
    If env var PROJECT_ROOT is set, it wins.
    """
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]  # <root>/src/config.py → <root>

def _first_existing(paths: list[Path]) -> Path | None:
    """
    Return the first existing path from a list, else None.
    Useful when you provide multiple fallbacks for a file.
    """
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None

# Current Area of Interest (e.g., "huambo", "moxico-leste")
AOI: str = os.environ.get("AOI", "moxico").lower().replace(" ", "-")

def _fmt(s: str) -> str:
    """Inject {AOI} into filename templates."""
    return s.format(AOI=AOI)


# ======================================================================
# 3) Paths (inputs/outputs), AOI-aware filenames
# ======================================================================

@dataclass(frozen=True)
class Paths:
    """Container for commonly used directories and files (AOI-aware)."""

    # Folders
    ROOT: Path
    DATA: Path
    VEC: Path
    RAS: Path
    OUT: Path
    OUT_R: Path
    OUT_T: Path
    OUT_F: Path

    # Rasters (already clipped to AOI)
    TRAVEL: Path
    POP: Path
    NTL: Path
    VEG: Path
    DROUGHT: Path
    FLOOD: Path
    RWI: Path | None

    # Vectors
    BND_ADM1: Path
    BND_ADM2: Path
    ROADS: Path
    RAIL: Path
    CROPLAND: Path
    ELEC: Path
    SETTLE: Path
    SITES: Path

    # Admin2 (RAPP) themes & outputs
    MUNI_DIR: Path
    MUNI_CLEAN_TBL: Path
    MUNI_CORR_TBL: Path
    MUNI_PROFILE_TBL: Path


def _build_paths() -> Paths:
    root = _detect_project_root()
    data = root / "data"
    ras  = data / "rasters"
    vec  = data / "vectors"
    out  = root / "outputs"
    out_r = out / "rasters"
    out_t = out / "tables"
    out_f = out / "figs"

    # Ensure outputs exist
    for p in (out, out_r, out_t, out_f):
        p.mkdir(parents=True, exist_ok=True)

    # Where Admin2 RAPP shapefiles live (per theme)
    muni_dir = vec

    return Paths(
        ROOT=root,
        DATA=data, VEC=vec, RAS=ras,
        OUT=out, OUT_R=out_r, OUT_T=out_t, OUT_F=out_f,

        # --- Rasters (AOI-parametric filenames) ---
        TRAVEL = ras / _fmt("ago_phy_{AOI}_traveltime_market.tif"),
        POP    = ras / _fmt("ago_pop_{AOI}_2025_CN_1km_R2025A_v1.tif"),
        NTL    = ras / _fmt("ago_phy_{AOI}_viirs_ntl_2024.tif"),
        VEG    = ras / _fmt("ago_phy_{AOI}_vegindex_mean_2024.tif"),  # 0.001..1
        DROUGHT= ras / _fmt("ago_phy_{AOI}_asishdfc_all_al30_2024.tif"),
        FLOOD  = ras / _fmt("ago_nhr_{AOI}_pluvialdefended_100rp_2020.tif"),
        RWI    = ras / _fmt("ago_pop_{AOI}_rwi_meta_2022.tif"),  # optional equity layer

        # --- Vectors (AOI-parametric filenames) ---
        BND_ADM1 = vec / _fmt("ago_bnd_{AOI}_adm1_a.shp"),
        BND_ADM2 = vec / _fmt("ago_bnd_{AOI}_adm2_a.shp"),
        ROADS    = vec / _fmt("ago_trs_{AOI}_roads_osm_l.shp"),
        RAIL     = vec / _fmt("ago_trs_{AOI}_railways_osm_l.shp"),
        CROPLAND = vec / _fmt("ago_phy_{AOI}_cropland_10m_worldcover_a.shp"),
        ELEC     = vec / _fmt("ago_pop_{AOI}_electricity_type_a.shp"),  # FinalElecCode2020: 1 grid, 99 unelectrified
        SETTLE   = vec / _fmt("ago_pop_{AOI}_settlement_type_a.shp"),   # IsUrban: 0 rural, 2 urban
        SITES    = vec / _fmt("ago_poi_{AOI}_projectloc_dm_p.shp"),

        # --- Admin2 RAPP themes & outputs ---
        MUNI_DIR = muni_dir,
        MUNI_CLEAN_TBL   = out_t / f"{AOI}_municipality_indicators.csv",
        MUNI_CORR_TBL    = out_t / f"{AOI}_corr_with_rural_poverty.csv",
        MUNI_PROFILE_TBL = out_t / f"{AOI}_municipality_profiles.csv",
    )

PATHS = _build_paths()


# ======================================================================
# 4) Resampling helper (centralized policy)
# ======================================================================

_RESAMPLING_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
    "average": Resampling.average,
    "mode": Resampling.mode,
    "max": Resampling.max,
    "min": Resampling.min,
    "med": Resampling.med,
    "q1": Resampling.q1,
    "q3": Resampling.q3,
    "sum": Resampling.sum,
    "rms": Resampling.rms,
}

def RESAMPLE(method: str | Resampling) -> Resampling:
    """
    Normalize a resampling spec to rasterio.enums.Resampling.
    Accepts either a string like "bilinear"/"nearest" or an enum already.
    """
    if isinstance(method, Resampling):
        return method
    key = str(method).lower().strip()
    try:
        return _RESAMPLING_MAP[key]
    except KeyError:
        raise ValueError(f"Unknown resampling method: {method!r}")

RESAMPLE_DEFAULT_CONT = Resampling.bilinear   # for continuous/ratio rasters
RESAMPLE_DEFAULT_CAT  = Resampling.nearest    # for categorical/masks

def RESAMPLE_KIND(kind: str) -> Resampling:
    """
    Map 'continuous'/'cat' to a consistent Resampling policy.
    """
    k = str(kind).lower().strip()
    if k in ("cont", "continuous"): return RESAMPLE_DEFAULT_CONT
    if k in ("cat", "categorical", "mask"): return RESAMPLE_DEFAULT_CAT
    raise ValueError(f"Unknown resampling kind: {kind!r}")

# Native (high-res) RP100 max depth. If provided, Step 04 will build a 1-km
# coverage fraction (share of subpixels >= FLOOD_DEPTH_RISK) on the fly.
FLOOD_NATIVE_TIF = ""   # e.g., PATHS.RAW / f"{AOI}_flood_rp100_maxdepth_30m.tif"

# Road rasterization strictness
ROAD_ALL_TOUCHED = False          # stricter default
FLOOD_EXCEED_FRACTION_MIN = 0.25  # fraction of 1-km cell that must be flooded


# ======================================================================
# 5) Admin2 (RAPP) ingestion controls & themes
# ======================================================================

# --- Admin2 fields (RAPP shapefiles) ---
ADMIN2_CODE_FIELD   = "ADM2CD_c"
ADMIN2_NAME1_FIELD  = "NAM_1"
ADMIN2_NAME2_FIELD  = "NAM_2"

# File prefixes for RAPP shapes (both gov & pop sources are supported)
MUNI_FILE_PREFIXES = ("ago_gov", "ago_pop")

# Skip themes entirely (if missing in your bundle). Examples present in THEME_VARS.
MUNI_THEMES_SKIP: tuple[str, ...] = ("climevents",)  # set to () to include all

# If True: quietly skip missing themes; if False: warn once per missing theme.
MUNI_SKIP_MISSING: bool = True

# Performance/verbosity knobs for Step 06
MUNI_SKIP_RASTERIZE: bool = False  # True: quick pass (tables only)
MUNI_LIMIT_THEMES: tuple[str, ...] = ()  # e.g., ("poverty", "traveltime") for fast dev
MUNI_COMPRESS_WIDE: bool = True    # write {AOI}_municipality_indicators.csv.gz

def muni_glob_for_theme(theme: str) -> list[str]:
    """
    Return glob patterns for Admin2 RAPP shapes that work with filenames like:
      ago_pop_{AOI}_adm2_{theme}_rapp_2020_a.shp
    Includes fallbacks to tolerate year/suffix differences.
    """
    tt = str(theme).lower()
    return [
        f"ago_*_{AOI.lower()}_adm2_{tt}_rapp_*.shp",  # AOI + adm2 token + theme
        f"ago_*_*_adm2_{tt}_rapp_*.shp",              # any AOI with adm2 token
        f"ago_*_{tt}_rapp_*.shp",                     # older naming
        f"*_{tt}_rapp_*.shp",
    ]

def muni_path_for(aoi_or_adm2: str, theme: str, return_first: bool = False) -> list[Path] | Path | None:
    """
    Build candidate exact paths for a given AOI/ADM2 + theme.

    Examples of matched filenames:
      <MUNI_DIR>/{pfx}_{aoi_or_adm2}_adm2_{theme}_rapp_2020_a.shp
    where pfx ∈ {"ago_gov", "ago_pop"}.

    Parameters
    ----------
    aoi_or_adm2 : str
        Province/ADM2 token used in the RAPP filenames (e.g., "huambo").
    theme : str
        Theme token (e.g., "poverty", "traveltime", "foodinsecurity").
    return_first : bool, default False
        If True, return the first path that exists (or None if none exist).
        If False (default), return the full candidate list (no existence check).

    Returns
    -------
    list[Path] | Path | None
        - When return_first=False (default): list of candidate Paths (may not exist).
        - When return_first=True: the first existing Path, or None if none found.
    """
    tt = str(theme).lower()
    a = str(aoi_or_adm2).lower()
    candidates = [
        PATHS.MUNI_DIR / f"{pfx}_{a}_adm2_{tt}_rapp_2020_a.shp"  # common case
        for pfx in MUNI_FILE_PREFIXES
    ]
    if not return_first:
        return candidates
    return _first_existing(candidates)


def muni_first_existing_path_for(aoi_or_adm2: str, theme: str) -> Path | None:
    """
    Convenience wrapper: return the first existing Admin2 RAPP shapefile for
    (aoi_or_adm2, theme), or None if none is found.
    """
    return muni_path_for(aoi_or_adm2, theme, return_first=True)


# RAPP schema & units
MUNI_JOIN_KEY = "ADM2CD_c"   # stable Admin2 key in RAPP (fallback to NAM_2 if missing)
RAPP_PCT_IS_0_100 = True     # RAPP percentages are 0–100 (True) or already 0–1 (False)

# Theme → friendly variable names
THEME_VARS: dict[str, dict[str, str]] = {
    "waterresources": {
        "data1": "rivers", "data2": "streams", "data3": "lakes",
        "data4": "lagoons", "data5": "wells",
    },
    "communications": {
        "data1": "telephone", "data2": "internet", "data3": "newspaper",
        "data4": "radio", "data5": "television", "data6": "none",
    },
    "infra": {
        "data1": "electricity", "data2": "water_storage", "data3": "veterinarians",
        "data4": "banking", "data5": "mech_agri_equip", "data6": "agri_schools",
        "data7": "primary_schools", "data8": "field_schools",
        "data9": "health_units", "data10": "agri_stock",
    },
    "foodinsecurity": {
        "data1": "went_without_food", "data2": "unable_eat_healthy",
        "data3": "few_types_of_food", "data4": "skipped_meal",
        "data5": "ate_less_than_needed", "data6": "ran_out_of_food",
        "data7": "hungry_did_not_eat", "data8": "without_food_all_day",
        "data9": "food_insec_scale",
    },
    "outflow": {
        "data1": "difficult_access_to_village", "data2": "insufficient_transport",
        "data3": "lack_conservation_infra", "data4": "high_transport_cost",
        "data5": "lack_transport_means", "data6": "other", "data7": "none",
    },
    "poverty": {
        "data1": "poverty_rural", "data2": "poverty_urban", "data3": "poverty_total",
    },
    "productions": {
        "data1": "diff_access_land", "data2": "unavailable_agri_land",
        "data3": "diff_access_water", "data4": "rural_exodus",
        "data5": "diff_dispose_products", "data6": "lack_rain",
        "data7": "lack_agri_equipment", "data8": "lack_tech_assist",
        "data9": "lack_manpower", "data10": "diff_access_credit",
    },
    "traveltime": { "data1": "avg_hours_to_market_financial" },
    "climevents": {
        "data1": "prolonged_drought", "data2": "drought", "data3": "strong_winds",
        "data4": "excessive_rainfall", "data5": "floods",
    },
}
ADMIN2_THEMES = tuple(THEME_VARS.keys())

# ------------------------------------------------------------------
# Derived travel-time semantics (RAPP → minutes)
# ------------------------------------------------------------------
# Raw RAPP shapefiles store average travel time in HOURS via the field
# "avg_hours_to_market_financial" (THEME_VARS["traveltime"]["data1"]).
# Step 06 converts this indicator to MINUTES and writes it under a
# minutes-based name so that file names and units match.
MUNI_TT_FIELD_MINUTES = "avg_minutes_to_market_financial"
MUNI_TT_WIDE_COL = f"traveltime__{MUNI_TT_FIELD_MINUTES}"

# A small featured set you can chart quickly or plug into priority (Step 07).
FEATURED_VARS: list[tuple[str, str]] = [
    ("poverty", "poverty_rural"),
    ("foodinsecurity", "food_insec_scale"),
    ("traveltime", "avg_hours_to_market_financial"),
    ("infra", "electricity"),
    ("outflow", "high_transport_cost"),
    ("climevents", "prolonged_drought"),
]


# ======================================================================
# 6) Optional overlays & references (may be missing for some AOIs)
# ======================================================================

# Optional grid overlays produced elsewhere (NOT used by Step 06).
# Keys are aliases used in Step 07; values are base filenames fed to out_r()
# (NO .tif extension).
OPTIONAL_GRID_OVERLAYS = {
    "rwi": "rwi_meta_1km",  # e.g., if Step 00 writes this
    # "pov": "muni_poverty_poverty_rural_1km",  # example if you prefer muni raster
}

OPTIONAL_GRID_OVERLAYS.update({
    "poverty":  "muni_poverty_poverty_rural_1km",
    "food":     "muni_foodinsecurity_food_insec_scale_1km",
    # Municipal travel-time to market, in MINUTES (post-Step-06 conversion)
    "muni_tt":  f"muni_traveltime_{MUNI_TT_FIELD_MINUTES}_1km",
})

# Optional AOI boundary polygon (for masking rasters, if used)
AOI_BND = PATHS.BND_ADM1

# If True, downstream steps may mask outputs strictly to the AOI boundary.
# (Keep as a simple module-level toggle so it’s readable without touching PARAMS.)
MASK_OUTSIDE_AOI: bool = False

# --- AOI masking policy -------------------------------------------------
# "defer"  : (recommended) Do NOT mask in Step 00; apply mask only at final writes in later steps.
# "eager"  : Mask at every write_gtiff_masked call (legacy behavior).
# "none"   : Never mask (ignore AOI).
MASK_POLICY: str = os.environ.get("MASK_POLICY", "defer").lower()  # "defer" | "eager" | "none"

# If True, we validate AOI before masking and skip mask (with warning) if no overlap.
AOI_BND_VALIDATE: bool = True

# Consider AOI "non-overlapping" if the mask covers less than this fraction of the target grid.
# Use a tiny floor (e.g., 0.005 = 0.5%) to still allow very small AOIs.
AOI_MIN_OVERLAP_PCT: float = 0.005

# Optional project/reference layers (steps should handle missing files gracefully)
PROJECTS_GOV = PATHS.VEC / f"ago_poi_{AOI}_projects_gov_p.shp"
PROJECTS_WB  = PATHS.VEC / f"ago_poi_{AOI}_projects_wb_p.shp"
PROJECTS_OTH = PATHS.VEC / f"ago_poi_{AOI}_projects_others_p.shp"

RAIL_STATIONS = PATHS.VEC / f"ago_trs_{AOI}_stations_osm_p.shp"
REFERENCE_POINTS = [
    {"name": "caala_lp", "lon": 15.56, "lat": -12.85},
    # {"name": "lobito_port", "lon": 13.559, "lat": -12.364},
]


# ======================================================================
# 7) Parameters / knobs (frozen dataclass)
# ======================================================================

def _sanitize_radii(vals) -> tuple[int, ...]:
    """
    Normalize a list/tuple of radii (km) to a sorted, unique, positive tuple.

    - If vals is None or empty, fall back to (5, 10, 30).
    - Any non-positive values are dropped.
    """
    if vals is None:
        return (5, 10, 30)
    cleaned = {int(r) for r in vals if int(r) > 0}
    return tuple(sorted(cleaned)) or (5, 10, 30)

@dataclass(frozen=True)
class Params:
    """All knobs in one place for repeatability."""

    # --- Required (non-default) ---
    TARGET_GRID: Path
    ISO_THRESH: tuple[int, ...]
    # Legacy priority weights (Step 03 compatibility)
    W_ACC: float
    W_POP: float
    W_VEG: float
    W_NTL: float
    W_DRT: float
    # Flood screening (RP100 depth threshold in meters)
    FLOOD_DEPTH_RISK: float
    # Beneficiary calc (households)
    PERSONS_PER_HH: float

    # --- Revised priority (Step 07) tunables ---
    # Component toggles (ACCESS, POP, VEG, NTL, DROUGHT)
    USE_COMPONENTS: tuple[int, int, int, int, int]
    # Masks
    MASK_REQUIRE_RURAL: bool
    MASK_MIN_CROPLAND: float   # 0 disables (e.g., 0.05 = 5%)
    # Transform caps
    NTL_CAP: float             # cap NTL before scaling
    VEG_MIN: float             # floor below which veg → 0
    # Smoothing & clustering
    SMOOTH_RADIUS: int         # 0 none; 1=3x3; 2=5x5
    MIN_CLUSTER_CELLS: int     # drop blobs smaller than this
    # Selection: choose one
    TOP_PCT_CELLS: float | None
    TOP_KM2: float | None
    # radii in km for site/cluster counts
    SYNERGY_RADII_KM: tuple[int, ...]    # e.g., (5, 10, 30)
    
    # --- Overlay weights (used only if overlays present) ---
    W_POV: float               # poverty (0..1)
    W_FOOD: float              # food insecurity (0..1)
    W_MTT: float               # admin2 travel time (minutes, inverted)
    W_RWI: float               # Meta RWI (-2..2 → 0..1)

    # --- OD-Lite gravity controls (v2) ---
    OD_ALPHA: float
    OD_GAMMA: float
    OD_F: str                  # "exp" or "pow"
    OD_LAMBDA: float           # exp impedance
    OD_BETA: float             # pow impedance
    OD_TRIPS_TOTAL: float
    OD_MAX_DIST_KM: float | None
    OD_USE_DOUBLY_CONSTRAINED: bool
    OD_N_AGENTS: int

    # Equity mass tilt for OD (and optionally elsewhere)
    USE_RWI_IN_MASS: bool
    RWI_WEIGHT: float

    # Legacy/compat OD flags (kept so older code doesn’t break)
    OD_MAX_LINES: int
    OD_USE_TRAVELTIME: bool

    # --- Defaulted (must come after non-defaults) ---
    # Road filtering policy:
    #   - None → use ALL OSM fclass values (no filter)
    #   - tuple(...) → allow-list of OSM fclass to keep
    ROAD_CLASSES_KEEP: tuple[str, ...] | None = None


PARAMS = Params(
    TARGET_GRID=PATHS.TRAVEL,
    ISO_THRESH=(30, 60, 120, 240),

    # Legacy weights (Step 03). Step 07 uses tunables below.
    W_ACC=0.35, W_POP=0.25, W_VEG=0.20, W_NTL=0.10, W_DRT=0.10,

    FLOOD_DEPTH_RISK=0.3,  # meters

    # Roads policy — OPTION A: use ALL OSM fclass (leave as None)
    ROAD_CLASSES_KEEP=None,
    # Roads policy — OPTION B: allow-list examples (uncomment to enable)
    # ROAD_CLASSES_KEEP=(
    #     "motorway","trunk","primary","secondary","tertiary",
    #     "motorway_link","trunk_link","primary_link","secondary_link","tertiary_link"
    # ),

    PERSONS_PER_HH=5.0,

    # Revised priority defaults (rural AOIs like Moxico)
    USE_COMPONENTS=(1, 1, 0, 0, 1),  # ACCESS, POP, VEG, NTL, DROUGHT
    MASK_REQUIRE_RURAL=True,
    MASK_MIN_CROPLAND=0.05,
    NTL_CAP=0.20,
    VEG_MIN=0.40,
    SMOOTH_RADIUS=1,
    MIN_CLUSTER_CELLS=30,
    TOP_PCT_CELLS=0.10,  # Top 10%
    TOP_KM2=None,        # or set e.g. 1200.0 for fixed area

    # Synergy / proximity
    SYNERGY_RADII_KM=_sanitize_radii((5, 10, 30)),

    # Overlay weights (used if overlays exist)
    W_POV=0.15,
    W_FOOD=0.10,
    W_MTT=0.10,
    W_RWI=0.15,

    # OD-Lite gravity controls (v2)
    OD_ALPHA=1.0,
    OD_GAMMA=1.0,
    OD_F="exp",
    OD_LAMBDA=0.015,
    OD_BETA=1.5,
    OD_TRIPS_TOTAL=1_000_000.0,
    OD_MAX_DIST_KM=1500.0,     # None to disable cutoff
    OD_USE_DOUBLY_CONSTRAINED=False,
    OD_N_AGENTS=50_000,

    # Equity tilt for OD masses
    USE_RWI_IN_MASS=True,
    RWI_WEIGHT=0.25,

    # Legacy/compat OD flags
    OD_MAX_LINES=200,
    OD_USE_TRAVELTIME=False,
)


# ======================================================================
# 8) Output helpers & canonical names
# ======================================================================

def log_denominators(log, *,
                     pop_total: float | None = None,
                     crop_total_km2: float | None = None,
                     elec_total_cells: float | None = None,
                     cell_area_km2: float | None = None,
                     prefix: str = "Denominators") -> None:
    """
    Emit one tidy INFO log line with optional pieces. Keeps spreadsheets & logs consistent.
    """
    parts: list[str] = []
    if pop_total is not None:
        parts.append(f"pop_total={int(pop_total):,}")
    if crop_total_km2 is not None:
        parts.append(f"cropland_total_km2={crop_total_km2:.2f}")
    if elec_total_cells is not None:
        parts.append(f"electrified_cells_total={int(elec_total_cells):,}")
    if cell_area_km2 is not None:
        parts.append(f"cell_area_km2={cell_area_km2:.3f}")
    if parts:
        log.info("%s | %s", prefix, " | ".join(parts))

def write_geo_sidecar(geotiff_path: Path, *, like=None, crs=None, transform=None,
                      aoi: str | None = AOI, extra: dict | None = None) -> None:
    """
    Save a light JSON next to a GeoTIFF with CRS/transform/shape.
    Pass either:
      - like=xarray.DataArray with .rio.crs/.rio.transform() and .shape, OR
      - crs / transform explicitly (rasterio-style Affine), OR
      - nothing (we'll open the GeoTIFF to read metadata).
    """
    meta = {"aoi": aoi, "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        if like is not None:
            meta.update({
                "crs": str(getattr(like.rio, "crs", None)),
                "transform": tuple(getattr(like.rio, "transform", lambda: None)()),
                "shape": tuple(getattr(like, "shape", (None, None))),
            })
        else:
            import rasterio as rio
            with rio.open(geotiff_path) as ds:
                meta.update({
                    "crs": str(ds.crs),
                    "transform": tuple(ds.transform),
                    "shape": (ds.height, ds.width),
                    "dtype": ds.dtypes[0] if ds.count > 0 else None,
                    "nodata": ds.nodata,
                })
        if crs is not None:       meta["crs"]       = str(crs)
        if transform is not None: meta["transform"] = tuple(transform)
        if extra:                 meta.update(extra)
    except Exception as e:
        meta["warning"] = f"sidecar capture encountered: {e!r}"

    side = Path(geotiff_path).with_suffix(Path(geotiff_path).suffix + ".geo.json")
    side.write_text(json.dumps(meta, indent=2))

def out_r(stem: str, ext: str = ".tif") -> Path:
    """Raster output path → outputs/rasters/{AOI}_{stem}.tif"""
    return PATHS.OUT_R / f"{AOI}_{stem}{ext}"

def out_t(stem: str, ext: str = ".csv") -> Path:
    """Table output path → outputs/tables/{AOI}_{stem}.csv"""
    return PATHS.OUT_T / f"{AOI}_{stem}{ext}"

def out_f(stem: str, ext: str = ".png") -> Path:
    """Figure output path → outputs/figs/{AOI}_{stem}.png"""
    return PATHS.OUT_F / f"{AOI}_{stem}{ext}"

# Admin2 precompute (Step 00)
ADMIN2_ID_TIF   = out_r("admin2_id_1km")      # int32 labels; 0 = background
ADMIN2_LUT_CSV  = out_t("admin2_lookup")      # columns: lab, ADM2CD_c, NAM_1, NAM_2

# Canonical filenames (kept for cross-step compatibility)
PRIORITY_TIF_V1         = out_r("priority_score_v1_0_1")
PRIORITY_TOP10_TIF_V1   = out_r("priority_top10_mask_v1")
PRIORITY_TIF            = out_r("priority_score_0_1")
PRIORITY_TOP10_TIF      = out_r("priority_top10_mask")
FLOOD1K_TIF             = out_r("flood_rp100_maxdepth_1km")
ROADS1K_TIF             = out_r("roads_main_1km")
ROADS_RISK_TIF          = out_r("roads_flood_risk_cells_1km")
ROADS_RISK_NEAR_TIF     = out_r("roads_flood_risk_near_priority_1km")
KPI_CSV                 = out_t("kpis_isochrones")
SITE_AUDIT_CSV          = out_t("site_audit_points")

# Site audit (Step 05)
SITE_AUDIT_RADIUS_CELLS = 5        # neighborhood radius in grid cells (5 km on 1-km grid)
SITE_ID_FIELD           = "no"     # attribute name used as site identifier in AGO POI shapefile

# Write small JSON sidecars (Steps 07 & 10)
WRITE_JSON_SIDECARS = True

# Shared/other outputs
CATCHMENTS_KPI_CSV      = out_t("catchments_kpis")
PRIORITY_CLUSTERS_TIF   = out_r("priority_clusters_1km")
MAP_TRAVEL_PRIORITY_PNG = out_f("map_travel_priority")

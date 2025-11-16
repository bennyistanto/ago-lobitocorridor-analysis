"""
Step 06 — Municipality (Admin2) ingest, normalization, correlations, rasterization

Purpose
-------
Bring in RAPP socio-economic indicators (by municipality) and make them easy to use:
1) Read all per-theme admin2 shapefiles (ago_*_{adm2}_{theme}_rapp_2020_a.shp)
2) Normalize attributes (percent → 0..1; hours → minutes)
3) Produce tidy tables & correlations vs rural poverty
4) Rasterize selected variables to the 1-km template grid (aligned with all other rasters)

Reads (from config)
-------------------
- PATHS.MUNI_DIR                : directory containing admin2 RAPP shapefiles
- THEME_VARS                    : dict mapping 'data1..N' → friendly var names for each theme
- ADMIN2_THEMES                 : tuple of theme names (keys of THEME_VARS)
- muni_glob_for_theme(), muni_path_for()  : helpers that build search patterns / exact paths
- PARAMS.TARGET_GRID            : template raster (travel time) → target grid for rasterization
- MUNI_JOIN_KEY                 : admin2 stable code column (fallback to NAM_2)
- RAPP_PCT_IS_0_100             : True if percent values are in 0..100
- FEATURED_VARS                 : small set of variables to guarantee rasterization
- MUNI_THEMES_SKIP              : themes to skip entirely (e.g., ("climevents",))
- MUNI_SKIP_MISSING             : if True, do not warn when a theme has no files
- MUNI_LIMIT_THEMES             : optional subset for fast dev cycles
- MUNI_SKIP_RASTERIZE           : if True, build tables only (skip rasters)

Writes
------
Tables (always plain CSV; no compression):
- PATHS.MUNI_CLEAN_TBL          : {AOI}_municipality_indicators.csv  (wide: one row per ADM2)
- PATHS.MUNI_CORR_TBL           : {AOI}_corr_with_rural_poverty.csv  (tidy: theme, var, r, p, n)
- PATHS.MUNI_PROFILE_TBL        : {AOI}_municipality_profiles.csv    (subset of headline vars + quintile)

Rasters (1-km, aligned to TARGET_GRID):
- outputs/rasters/{AOI}_muni_{theme}_{varname}_1km.tif     for selected variables

Notes
-----
- Percentages are normalized to 0..1 internally if RAPP_PCT_IS_0_100=True.
- Food insecurity scale (0..100) is also normalized to 0..1 for consistency.
- Traveltime hours are converted to minutes (float).
- Correlations are computed only where rural poverty exists (rowwise).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize

from time import perf_counter

from config import (
    AOI, PATHS, PARAMS, THEME_VARS, ADMIN2_THEMES, FEATURED_VARS,
    MUNI_JOIN_KEY, RAPP_PCT_IS_0_100,
    muni_glob_for_theme, muni_path_for,
    out_r, get_logger,
    MUNI_THEMES_SKIP, MUNI_SKIP_MISSING,
    MUNI_SKIP_RASTERIZE, MUNI_LIMIT_THEMES,
    MUNI_COMPRESS_WIDE,
)

from utils_geo import open_template, write_gtiff_masked  # NOTE: we avoid rasterize_vector(vec=...) here

log = get_logger(__name__)


# ------------------------------ Helpers --------------------------------------


def _read_theme_layers(theme: str) -> gpd.GeoDataFrame:
    """
    Read the AOI-specific admin2 shapefile for a theme.
    Returns a GeoDataFrame with standardized columns; value fields still named 'dataN'.
    """
    from config import muni_first_existing_path_for, AOI, PATHS

    shp = muni_first_existing_path_for(AOI, theme)
    if shp is None or not Path(shp).exists():
        if MUNI_SKIP_MISSING:
            log.info("Skipping theme=%s: no AOI file found in %s.", theme, PATHS.MUNI_DIR)
            return gpd.GeoDataFrame(columns=["geometry"])
        log.warning("Theme=%s: AOI file not found (looked in %s).", theme, PATHS.MUNI_DIR)
        return gpd.GeoDataFrame(columns=["geometry"])

    try:
        gdf = gpd.read_file(shp)
    except Exception as e:
        log.warning("Failed reading %s: %s", shp, e)
        return gpd.GeoDataFrame(columns=["geometry"])

    # Ensure key fields exist (fallback to NAM_2 for code if needed)
    if "ADM2CD_c" not in gdf.columns:
        gdf["ADM2CD_c"] = gdf.get("NAM_2", pd.Series([None] * len(gdf)))
    return gdf


def _rename_data_columns(gdf: gpd.GeoDataFrame, theme: str) -> gpd.GeoDataFrame:
    """
    Rename data1..dataN → friendly variable names per THEME_VARS[theme].
    Unlisted dataN are dropped (keeps only mapped variables for that theme).
    """
    var_map: Dict[str, str] = THEME_VARS.get(theme, {})
    keep_cols = ["ADM2CD_c", "NAM_2", "NAM_1", "geometry"]

    # Build rename dict only for existing data* columns
    ren: Dict[str, str] = {raw: nice for raw, nice in var_map.items() if raw in gdf.columns}

    cols = keep_cols + list(ren.keys())
    cols = [c for c in cols if c in gdf.columns]
    gdf2 = gdf[cols].rename(columns=ren)
    gdf2["theme"] = theme
    return gdf2


def _normalize_values(df: pd.DataFrame, theme: str) -> pd.DataFrame:
    """
    Normalize columns:
    - Percentages: if RAPP_PCT_IS_0_100=True, divide by 100 into 0..1
    - Food insecurity scale: also divide by 100 → 0..1
    - Traveltime hours: convert to minutes (float)
    """
    ignore = {"ADM2CD_c", "NAM_2", "NAM_1", "theme", "geometry"}
    var_cols = [c for c in df.columns if c not in ignore and df[c].dtype != object]

    # Traveltime: hours → minutes
    if theme == "traveltime":
        for c in var_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") * 60.0
        log.info("Converted traveltime theme hours→minutes for %s (columns: %s)", theme, ", ".join(var_cols))
        return df
    
    # Percent → 0..1
    if RAPP_PCT_IS_0_100:
        for c in var_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df


def _theme_var_cols(df: pd.DataFrame, theme: str) -> List[str]:
    """
    Return the friendly variable columns for `theme` that actually exist in df.
    """
    var_map = THEME_VARS.get(theme, {})
    wanted = list(var_map.values())  # friendly names
    return [c for c in wanted if c in df.columns]


def _wide_table(all_themes_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Pivot per-theme frames into a single WIDE table, one row per ADM2, columns as "<theme>__<var>".
    """
    if all_themes_gdf.empty:
        return pd.DataFrame(columns=["ADM2CD_c", "NAM_2", "NAM_1"])

    # Don't assume a geometry column exists (gdf_all_tables has none)
    df = pd.DataFrame(all_themes_gdf.copy())
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])
    if "ADM2CD_c" not in df.columns:
        df["ADM2CD_c"] = df.get("NAM_2", pd.Series([None]*len(df)))

    parts: List[pd.DataFrame] = []
    for theme in sorted(all_themes_gdf["theme"].unique()):
        sub = df[df["theme"] == theme].copy()
        id_cols = ["ADM2CD_c", "NAM_2", "NAM_1"]
        var_cols = [c for c in sub.columns if c not in (id_cols + ["theme"])]

        sub = sub[id_cols + var_cols]
        sub = sub.rename(columns={c: f"{theme}__{c}" for c in var_cols})
        parts.append(sub)

    out = None
    for sub in parts:
        out = sub if out is None else out.merge(sub, on=["ADM2CD_c", "NAM_2", "NAM_1"], how="outer")

    return out


def _corr_vs_poverty(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Pearson correlation with 'poverty__poverty_rural' across municipalities.
    Returns tidy DataFrame: theme, var, r, p, n.
    """
    from scipy.stats import pearsonr

    targ_col = "poverty__poverty_rural"
    if targ_col not in wide.columns:
        miss = [c for c in wide.columns if c.startswith("poverty__")]
        log.warning(
            "Rural poverty column '%s' not found; skipping correlations. "
            "Poverty columns present: %s",
            targ_col, miss[:5]
        )
        return pd.DataFrame(columns=["theme", "var", "r", "p", "n"])

    w = wide.copy()
    w[targ_col] = pd.to_numeric(w[targ_col], errors="coerce")
    w = w[~w[targ_col].isna()].copy()
    if w.empty:
        log.warning("All rows have NaN in %s; skipping correlations.", targ_col)
        return pd.DataFrame(columns=["theme", "var", "r", "p", "n"])

    theme_prefixes = tuple(t + "__" for t in ADMIN2_THEMES)
    candidates = [c for c in w.columns if c.startswith(theme_prefixes) and c != targ_col]

    x = w[targ_col].astype(float)
    results: List[Tuple[str, str, float, float, int]] = []
    for col in candidates:
        y = pd.to_numeric(w[col], errors="coerce")
        m = ~x.isna() & ~y.isna()
        n = int(m.sum())
        if n < 3:
            continue
        try:
            r, p = pearsonr(x[m].values.astype(float), y[m].values.astype(float))
            theme, var = col.split("__", 1)
            results.append((theme, var, r, p, n))
        except Exception as e:
            log.debug("pearsonr failed for %s vs %s: %s", targ_col, col, e)

    out = pd.DataFrame(results, columns=["theme", "var", "r", "p", "n"])
    if not out.empty:
        out = out.sort_values(by="r", ascending=False)
    return out


def _rasterize_selected_vars(gdf_all: gpd.GeoDataFrame, T) -> List[Path]:
    """
    Polygon-burn rasterization: for each (theme, variable), burn a constant value per Admin2 polygon
    onto the 1-km template grid. Outside AOI is written as NaN via write_gtiff_masked().
    Returns list of written file paths.
    """
    written: List[Path] = []
    if gdf_all.empty or "geometry" not in gdf_all.columns:
        log.warning("No geometries available; skipping rasterization.")
        return written

    # Align CRS to the template
    try:
        if gdf_all.crs is not None and T.rio.crs is not None and gdf_all.crs != T.rio.crs:
            gdf_all = gdf_all.to_crs(T.rio.crs)
    except Exception as e:
        log.warning("Failed to align CRS for rasterization: %s", e)

    # Determine which themes we actually ingested
    themes_present = sorted(set(gdf_all["theme"])) if "theme" in gdf_all.columns else []

    # Rasterize all variables that exist for each theme
    for theme in themes_present:
        sub = gdf_all[gdf_all["theme"] == theme].copy()
        if sub.empty:
            continue

        # Pick only the friendly variable columns that are present
        vars_here = _theme_var_cols(sub, theme)
        if not vars_here:
            log.info("Theme=%s: no mapped variables present; skipping burn.", theme)
            continue

        # Clean geometry rows
        sub = sub[~sub.geometry.is_empty & sub.geometry.notna()].copy()
        if sub.empty:
            log.info("Theme=%s: empty geometry after cleaning; skipping.", theme)
            continue

        # For reproducibility, ensure we have exactly one row per ADM2 with numeric values
        # (If multiple rows exist per ADM2, we take mean per variable—same as tables.)
        keys = ["ADM2CD_c", "NAM_1", "NAM_2", "geometry"]
        numcols = [c for c in vars_here]
        for c in numcols:
            sub[c] = pd.to_numeric(sub[c], errors="coerce")

        grp = (
            sub
            .dropna(subset=["ADM2CD_c"])
            .groupby(keys, as_index=False)[numcols]
            .mean()
        )

        # For each variable, burn constant value per polygon
        out_shape = (T.sizes["y"], T.sizes["x"])
        tf = T.rio.transform()

        for var in vars_here:
            # Build shapes list excluding NaN values
            shapes = [(geom, float(val)) for geom, val in zip(grp.geometry, grp[var]) if np.isfinite(val)]
            if not shapes:
                log.info("Skip rasterize: theme=%s var=%s (no finite values).", theme, var)
                continue

            arr = rasterize(
                shapes=shapes,
                out_shape=out_shape,
                transform=tf,
                fill=np.nan,
                all_touched=False,        # conservative; avoids halo/speckle
                dtype="float32",
            )

            R = T.copy(deep=False)
            R = R.where(False, np.nan)
            R.values[:] = arr

            out_path = out_r(f"muni_{theme}_{var}_1km")
            write_gtiff_masked(R, out_path, like=T, nodata=np.nan)
            written.append(out_path)
            # quick stats for sanity
            vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan
            vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan
            log.info("Wrote %s | min=%.4f max=%.4f", out_path.name, vmin, vmax)

    return written


# ------------------------------ Main -----------------------------------------


def main() -> None:
    """
    Ingest admin2 socio-economic themes, normalize, write tidy tables,
    compute correlations vs rural poverty, and rasterize selected variables.
    """
    # Load target grid (1-km travel) only for CRS/transform/extent/shape
    T = open_template(PARAMS.TARGET_GRID)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(f"Target grid loaded | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}")

    # 1) Read, rename, normalize per theme; then concat
    frames: List[gpd.GeoDataFrame] = []

    # Build final theme list honoring both skip list and developer limit
    skip_set = set(MUNI_THEMES_SKIP or ())
    base_themes = [t for t in ADMIN2_THEMES if t not in skip_set]
    themes_to_run = [t for t in base_themes if (not MUNI_LIMIT_THEMES or t in set(MUNI_LIMIT_THEMES))]

    if skip_set:
        log.info("Step 06 will skip themes: %s", tuple(sorted(skip_set)))
    if MUNI_LIMIT_THEMES:
        log.info("Step 06 limited to themes: %s", tuple(sorted(set(MUNI_LIMIT_THEMES))))

    for theme in themes_to_run:
        t0 = perf_counter()
        gdf_raw = _read_theme_layers(theme)
        t1 = perf_counter()

        if gdf_raw.empty:
            log.info("Theme=%s: no data (%.2fs).", theme, t1 - t0)
            continue

        gdf = _rename_data_columns(gdf_raw, theme); t2 = perf_counter()
        gdf = _normalize_values(gdf, theme);         t3 = perf_counter()

        frames.append(gdf)
        log.info(
            "Theme=%s: read=%.2fs, rename=%.2fs, normalize=%.2fs, rows=%d, cols=%d",
            theme, (t1 - t0), (t2 - t1), (t3 - t2), len(gdf), len(gdf.columns)
        )

    if not frames:
        log.warning("No admin2 data ingested; nothing to write.")
        # still write empty tables to keep downstream stable
        pd.DataFrame().to_csv(PATHS.MUNI_CLEAN_TBL, index=False)
        pd.DataFrame().to_csv(PATHS.MUNI_CORR_TBL, index=False)
        pd.DataFrame().to_csv(PATHS.MUNI_PROFILE_TBL, index=False)
        return

    gdf_all = pd.concat(frames, ignore_index=True)
    log.info(f"Ingested admin2 themes: {sorted(gdf_all['theme'].unique())} | rows={len(gdf_all)}")

    # --- NEW: collapse duplicates per ADM2+theme for TABLES ONLY (keep full gdf_all for rasters) ---
    keys = ["ADM2CD_c", "NAM_1", "NAM_2", "theme"]
    non_group_cols = keys + ["geometry"]

    # numeric-only list (coerce to numeric so mean() behaves)
    numcols = [c for c in gdf_all.columns if c not in non_group_cols]
    gdf_all_num = gdf_all.copy()
    for c in numcols:
        gdf_all_num[c] = pd.to_numeric(gdf_all_num[c], errors="coerce")

    # Aggregate to one record per ADM2 x theme (mean of numeric fields)
    gdf_all_tables = (
        gdf_all_num
        .dropna(subset=["ADM2CD_c"])
        .groupby(keys, as_index=False)[numcols]
        .mean()
    )

    # We’ll use gdf_all_tables to build the wide/profiles CSVs;
    # and keep original gdf_all (with full geometry) for rasterization.

    # 2) Build WIDE table per ADM2 (columns '<theme>__<var>') from aggregated copy
    wide = _wide_table(gdf_all_tables)

    # Ensure one row per ADM2 and remove all-empty columns
    wide = (
        wide
        .dropna(axis=1, how="all")
        .drop_duplicates(subset=["ADM2CD_c", "NAM_1", "NAM_2"])
        .sort_values(["NAM_1", "NAM_2"], ignore_index=True)
    )

    # --- Write plain CSV (no gzip) ---
    wide.to_csv(PATHS.MUNI_CLEAN_TBL, index=False)
    log.info("Wrote %s | rows=%d | cols=%d", PATHS.MUNI_CLEAN_TBL.name, len(wide), len(wide.columns))

    # 3) Correlations vs rural poverty
    corr = _corr_vs_poverty(wide)
    corr.to_csv(PATHS.MUNI_CORR_TBL, index=False)
    log.info("Wrote %s | rows=%d", PATHS.MUNI_CORR_TBL.name, len(corr))

    # 4) Profiles (lean fallback builder + robust quintiles)
    base_cols = [c for c in ("ADM2CD_c", "NAM_1", "NAM_2") if c in wide.columns]
    keep_cols = base_cols + [c for c in wide.columns if c.startswith(("poverty__", "traveltime__"))]
    profiles = wide[keep_cols].copy()
    log.warning("Using fallback profile builder (no _profiles_table found).")

    # --- NEW: clean profiles upfront ---
    profiles = (
        profiles
        .dropna(axis=1, how="all")
        .drop_duplicates(subset=["ADM2CD_c", "NAM_1", "NAM_2"])
        .reset_index(drop=True)
    )

    col = "poverty__poverty_rural"
    if col in profiles.columns:
        dup_count = sum(c == col for c in profiles.columns)
        if dup_count > 1:
            log.warning("Multiple '%s' columns detected (%d). Using the first instance.", col, dup_count)
        q_raw = profiles.loc[:, col]
        if isinstance(q_raw, pd.DataFrame):
            q_raw = q_raw.iloc[:, 0]
        q_num = pd.to_numeric(q_raw, errors="coerce")
        valid = q_num.dropna()
        if valid.empty or valid.nunique() < 2:
            log.warning("Insufficient variation in %s to compute quintiles; skipping.", col)
        else:
            try:
                quints = pd.qcut(q_num.rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
                profiles["poverty_quintile"] = quints
            except Exception as e:
                log.warning("pd.qcut failed for %s (%s). Falling back to tertiles.", col, e)
                try:
                    tert = pd.qcut(q_num.rank(method="first"), 3, labels=[1, 2, 3], duplicates="drop")
                    profiles["poverty_quintile"] = tert
                except Exception as e2:
                    log.warning("Fallback tertiles also failed for %s (%s). Skipping quintiles.", col, e2)

    profiles.to_csv(PATHS.MUNI_PROFILE_TBL, index=False)
    log.info("Wrote %s | rows=%d", PATHS.MUNI_PROFILE_TBL.name, len(profiles))

    # 5) Rasterize selected variables (optional)
    if MUNI_SKIP_RASTERIZE:
        log.info("Skipping rasterization per config (MUNI_SKIP_RASTERIZE=True).")
        written = []
    else:
        written = _rasterize_selected_vars(gdf_all, T)
    log.info("Rasterized %d muni variables to 1-km grid", len(written))

    # --- sanity: ensure key overlays exist for downstream (Step 07/10) ---
    must_have = [
        out_r("muni_poverty_poverty_rural_1km"),
        out_r("muni_foodinsecurity_food_insec_scale_1km"),
        out_r("muni_traveltime_avg_hours_to_market_financial_1km"),
    ]
    for p in must_have:
        if not Path(p).exists():
            log.warning("Expected overlay missing → %s (check THEME_VARS mapping & rasterization)", Path(p).name)
        else:
            log.info("Overlay ready: %s", Path(p).name)


    log.info("Step 06 complete.")


if __name__ == "__main__":
    main()

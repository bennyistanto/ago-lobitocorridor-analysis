"""
Step 14 — OD-Lite: Gravity Matrix & Agent Sampling (Admin2 zones)

Purpose
-------
Construct an Origin–Destination (OD) matrix across zones (default: Admin2 municipalities)
and sample individual trips (agents) for light-weight mobility/interaction analysis.

Why this is useful
------------------
- Gives a corridor-wide sense of interaction intensity between municipalities (or grids).
- Supports planning questions like “which pairs of places exchange the most trips?”
- Output tables are directly plottable (desire lines, chord diagrams, sankeys, etc.).

Approach (default = Admin2 zones)
---------------------------------
1) Build zones:
   - Use your RAPP Admin2 polygons (auto-pick a theme file for this AOI).
   - Compute zone centroids (lon/lat).
2) Aggregate population:
   - Sum population from {AOI}_pop_1km.tif within each Admin2 polygon.
3) Deterrence (separations):
   - Compute geodesic distance (km) between zone centroids.
   - (Optional) use travel-time or any other generalized separation later.
4) Gravity model:
   - F_ij = K * (P_i^alpha) * (P_j^beta) * exp(-lambda * D_ij) * (W_ij^gamma)
     where W_ij is optional bilateral weight (defaults to 1).
   - K is a global factor to scale total flow to TRIPS_TOTAL.
5) Sampling (optional but recommended):
   - Multinomial sampling of N agents ~ normalized F_ij.
   - Output CSV of agents with origin/destination lon/lat (and zone IDs).

Inputs (expected)
-----------------
- PATHS.OUT_R/{AOI}_pop_1km.tif         : 1-km population raster (Step 00)
- Admin2 RAPP polygon for this AOI       : picked automatically from PATHS.MUNI_DIR
- (Optional) A bilateral weight matrix CSV (see CONFIG near top) [not required]

Outputs
-------
- outputs/tables/{AOI}_od_gravity.csv         : tidy OD matrix (i,j, flow, dist_km, P_i, P_j)
- outputs/tables/{AOI}_od_zone_attrs.csv      : zone attributes (id, code, names, lon, lat, pop)
- outputs/tables/{AOI}_od_agents.csv          : sampled trips (origin_lon, origin_lat, dest_lon, dest_lat, oi, dj)

Tuning (no code edits needed)
-----------------------------
Edit parameters here or move them to config.py later:
- ALPHA, BETA, LAMBDA, GAMMA : gravity exponents
- TRIPS_TOTAL                : total trips to scale matrix to
- N_AGENTS                   : number of agents to sample
- MIN_POP_ZONE               : drop tiny zones (population filter)
- USE_ADMIN2_ZONES           : True (default). If False, we can add grid zones later.

Notes / limits
--------------
- Uses centroid-to-centroid geodesic distance as separation (simple, transparent).
- If you later want travel-time based separation, we can add it via a skim (pairwise costs).
- This is *not* a network assignment; it’s an interaction model (Plan B “lite”).
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import Point
from pyproj import Geod

from config import (
    AOI, PATHS, PARAMS, get_logger, out_t, out_r,
    muni_path_for, ADMIN2_THEMES,
)

from utils_geo import open_template

log = get_logger(__name__)
_G = Geod(ellps="WGS84")

# -----------------------------------------------------------------------------
# Tunables (feel free to move these into config.Params later)
ALPHA: float = 1.0      # origin mass exponent
BETA: float  = 1.0      # destination mass exponent
LAMBDA: float = 0.05    # distance deterrence (per km) for exp(-lambda * d)
GAMMA: float  = 0.0     # bilateral weight exponent (kept 0.0 => disabled)

TRIPS_TOTAL: float = 200_000.0  # total system trips to scale the matrix
N_AGENTS: int = 25_000          # number of agents to sample from F_ij
MIN_POP_ZONE: float = 50.0      # drop zones with population below this threshold

USE_ADMIN2_ZONES: bool = True   # default; grid zones can be added later
# Optional: path to a bilateral weight CSV with columns [ADM2_i, ADM2_j, weight]
BILATERAL_WEIGHTS_CSV: Path | None = None
# Optional: use RWI to modulate zone "mass" (pop). Default OFF to keep parity.
USE_RWI_IN_MASS: bool = False
RWI_WEIGHT: float = 0.25  # 0..1; how much normalized (-RWI) tilts mass upward (equity tilt)

# -----------------------------------------------------------------------------


# ============================ helpers ============================

def _pick_admin2_geom() -> Path | None:
    """Pick an Admin2 shapefile for this AOI (prefer 'poverty' theme; else first available)."""
    pov = muni_path_for(AOI, "poverty")
    if pov.exists():
        return pov
    for theme in ADMIN2_THEMES:
        p = muni_path_for(AOI, theme)
        if p.exists():
            return p
    return None


def _geodesic_km(lon1, lat1, lon2, lat2) -> float:
    """Geodesic distance in km (WGS84 ellipsoid)."""
    _, _, dist_m = _G.inv(lon1, lat1, lon2, lat2)
    return float(dist_m) / 1000.0


def _pairwise_geodesic_km(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Compute full pairwise geodesic distance matrix (km) for zone centroids."""
    n = len(lons)
    D = np.zeros((n, n), dtype="float32")
    for i in range(n):
        # broadcast against all j
        _, _, dist_m = _G.inv(np.full(n, lons[i]), np.full(n, lats[i]), lons, lats)
        D[i, :] = (dist_m / 1000.0).astype("float32")
    return D


def _aggregate_pop_by_admin2(pop_1k: xr.DataArray, admin2_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Zonal sum of population by Admin2 polygons using rasterization + bincount.
    Returns DataFrame [ADM2_label, ADM2CD_c, NAM_1, NAM_2, pop].
    """
    # normalize CRS
    g = admin2_gdf.to_crs("EPSG:4326") if admin2_gdf.crs else admin2_gdf.set_crs("EPSG:4326")
    if "ADM2CD_c" not in g.columns or "NAM_1" not in g.columns or "NAM_2" not in g.columns:
        raise ValueError("Admin2 file missing required columns ADM2CD_c, NAM_1, NAM_2")

    # numeric labels 1..N
    codes = g["ADM2CD_c"].astype(str)
    lab_vals, uniques = pd.factorize(codes)
    g["ADM2_label"] = lab_vals + 1
    lut = pd.DataFrame({
        "ADM2_label": np.arange(1, len(uniques)+1, dtype=np.int64),
        "ADM2CD_c": uniques,
    }).merge(g[["ADM2_label","NAM_1","NAM_2"]].drop_duplicates(), on="ADM2_label", how="left")

    # rasterize labels (all_touched=True)
    labels = g[["ADM2_label", "geometry"]].rio.write_crs("EPSG:4326", inplace=True).rasterize(
        out_shape=pop_1k.rio.shape, transform=pop_1k.rio.transform(),
        all_touched=True, fill=0, dtype=np.int64
    )
    labels = labels.where(labels > 0, np.nan)

    # bincount sum over valid population cells
    lab = labels.values
    val = pop_1k.values
    m = np.isfinite(lab) & np.isfinite(val)
    lab1 = lab[m].astype(np.int64)
    val1 = val[m].astype(np.float64)
    sums = np.bincount(lab1, weights=val1, minlength=int(np.nanmax(lab))+1)

    df = lut.copy()
    df["pop"] = df["ADM2_label"].map(lambda k: float(sums[int(k)]) if int(k) < len(sums) else 0.0)
    return df


def _aggregate_mean_by_admin2(da_1k: xr.DataArray, admin2_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Zonal MEAN of a 1-km raster by Admin2 polygons using rasterized labels + bincount.
    Returns DataFrame [ADM2_label, ADM2CD_c, NAM_1, NAM_2, mean_val].
    """
    g = admin2_gdf.to_crs("EPSG:4326") if admin2_gdf.crs else admin2_gdf.set_crs("EPSG:4326")
    codes = g["ADM2CD_c"].astype(str)
    lab_vals, uniques = pd.factorize(codes)
    g["ADM2_label"] = lab_vals + 1

    labels = g[["ADM2_label", "geometry"]].rio.write_crs("EPSG:4326", inplace=True).rasterize(
        out_shape=da_1k.rio.shape, transform=da_1k.rio.transform(),
        all_touched=True, fill=0, dtype=np.int64
    )
    labels = labels.where(labels > 0, np.nan)

    lab = labels.values
    val = da_1k.values
    m = np.isfinite(lab) & np.isfinite(val)
    if not np.any(m):
        out = pd.DataFrame({"ADM2CD_c": uniques, "NAM_1": np.nan, "NAM_2": np.nan, "mean_val": np.nan})
        return out

    lab1 = lab[m].astype(np.int64)
    val1 = val[m].astype(np.float64)
    sums = np.bincount(lab1, weights=val1, minlength=int(np.nanmax(lab))+1)
    cnts = np.bincount(lab1, minlength=int(np.nanmax(lab))+1)
    means = np.divide(sums, np.where(cnts == 0, np.nan, cnts))

    lut = g[["ADM2_label","ADM2CD_c","NAM_1","NAM_2"]].drop_duplicates().copy()
    lut["mean_val"] = lut["ADM2_label"].map(lambda k: float(means[int(k)]) if int(k) < len(means) else np.nan)
    return lut.drop(columns=["ADM2_label"])


def _load_bilateral_weights(df_zones: pd.DataFrame) -> pd.DataFrame | None:
    """
    Optional: load a bilateral weights table with columns [ADM2_i, ADM2_j, weight] (string codes).
    Returns a square matrix dataframe aligned to df_zones order, or None if not provided.
    """
    if BILATERAL_WEIGHTS_CSV is None:
        return None
    p = Path(BILATERAL_WEIGHTS_CSV)
    if not p.exists():
        log.warning(f"Bilateral weights CSV not found: {p}")
        return None
    w = pd.read_csv(p)
    need = {"ADM2_i", "ADM2_j", "weight"}
    if not need.issubset(w.columns):
        log.warning(f"Bilateral weights CSV missing columns {need}; ignoring.")
        return None
    # pivot into a square table keyed by ADM2CD_c
    wide = w.pivot_table(index="ADM2_i", columns="ADM2_j", values="weight", fill_value=1.0)
    # align to df_zones order (ADM2CD_c)
    codes = df_zones["ADM2CD_c"].astype(str).tolist()
    wide = wide.reindex(index=codes, columns=codes, fill_value=1.0)
    # ensure ndarray
    wide = wide.astype("float32")
    return wide


def _sample_agents_from_matrix(F: np.ndarray, zone_lon: np.ndarray, zone_lat: np.ndarray, n_agents: int,
                               rng: np.random.Generator) -> pd.DataFrame:
    """
    Multinomial sampling of agents from OD matrix F (nonnegative).
    Returns DataFrame with origin/destination lon/lat and zone indices (oi, dj).
    """
    F = np.array(F, dtype="float64")
    F[F < 0] = 0.0
    S = F.sum()
    if S <= 0:
        return pd.DataFrame(columns=["origin_lon","origin_lat","dest_lon","dest_lat","oi","dj"])
    p = (F / S).ravel()
    # draw flat indices
    idx_flat = rng.choice(p.size, size=int(n_agents), replace=True, p=p)
    n = zone_lon.size
    oi = (idx_flat // n).astype("int32")
    dj = (idx_flat %  n).astype("int32")
    df = pd.DataFrame({
        "origin_lon": zone_lon[oi], "origin_lat": zone_lat[oi],
        "dest_lon":   zone_lon[dj], "dest_lat":   zone_lat[dj],
        "oi": oi, "dj": dj,
    })
    return df


# ============================ main ============================

def main() -> None:
    """
    Build an Admin2-based gravity OD matrix and sample agents.
    Writes three CSVs under outputs/tables for downstream viz.
    """
    if not USE_ADMIN2_ZONES:
        log.error("Only Admin2 zones are implemented in Step 14 for now. Set USE_ADMIN2_ZONES=True.")
        return

    # 1) Load template & population raster
    T = open_template(PARAMS.TARGET_GRID)
    pop_fp = PATHS.OUT_R / f"{AOI}_pop_1km.tif"
    if not pop_fp.exists():
        raise FileNotFoundError(f"Population raster not found: {pop_fp.name}")
    pop = rxr.open_rasterio(pop_fp, masked=True).squeeze()
    if (pop.shape != T.shape) or (pop.rio.transform() != T.rio.transform()) or (pop.rio.crs != T.rio.crs):
        pop = pop.rio.reproject_match(T, resampling="bilinear")

    # Optional RWI (Meta -2..+2), aligned to template
    rwi_fp = out_r("rwi_meta_1km")
    rwi = None
    if Path(rwi_fp).exists():
        rwi = rxr.open_rasterio(rwi_fp, masked=True).squeeze()
        if (rwi.shape != T.shape) or (rwi.rio.transform() != T.rio.transform()) or (rwi.rio.crs != T.rio.crs):
            rwi = rwi.rio.reproject_match(T, resampling="bilinear")

    # 2) Pick an Admin2 shapefile
    adm2_fp = _pick_admin2_geom()
    if adm2_fp is None:
        raise FileNotFoundError("No Admin2 RAPP shapefile found for this AOI in PATHS.MUNI_DIR.")
    g = gpd.read_file(adm2_fp)
    g = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
    g = g[~g.geometry.is_empty & g.geometry.notnull()].copy()
    need = {"ADM2CD_c", "NAM_1", "NAM_2"}
    if not need.issubset(g.columns):
        raise ValueError(f"{adm2_fp.name} missing required columns {need}")

    # 3) Aggregate population by Admin2
    df_z = _aggregate_pop_by_admin2(pop, g)

    # Optional: equity tilt with RWI
    if USE_RWI_IN_MASS and (rwi is not None):
        df_z = _merge_rwi_as_mass_tilt(df_z, rwi, weight=RWI_WEIGHT)

    # 4) Distances between zone centroids
    cent = g.copy()
    cent["centroid"] = cent.geometry.centroid
    cent["lon"] = cent["centroid"].x.astype("float64")
    cent["lat"] = cent["centroid"].y.astype("float64")
    D = _pairwise_geodesic_km(cent["lon"].values, cent["lat"].values)

    # 5) Gravity flows
    F = _gravity_matrix(
        pop_i=df_z["pop"].values,
        pop_j=df_z["pop"].values,
        D=D,
        alpha=ALPHA, beta=BETA, lam=LAMBDA, gamma=GAMMA,
        bilateral=None,  # plug weights here if you add a W_ij table
        trips_total=TRIPS_TOTAL,
    )

    # 6) Save outputs
    out_grav = PATHS.OUT_T / f"{AOI}_od_gravity.csv"
    out_z    = PATHS.OUT_T / f"{AOI}_od_zone_attrs.csv"
    pd.DataFrame({
        "oi": np.repeat(df_z.index.values, len(df_z)),
        "dj": np.tile(df_z.index.values, len(df_z)),
        "flow": F.ravel(),
        "dist_km": D.ravel(),
    }).to_csv(out_grav, index=False)
    df_z.assign(lon=cent["lon"].values, lat=cent["lat"].values).to_csv(out_z, index=False)
    log.info(f"Wrote {out_grav.name}, {out_z.name}")

    # 7) Sample agents
    agents = _sample_agents_from_flows(
        F, cent["lon"].values, cent["lat"].values, n_agents=N_AGENTS, seed=42
    )
    out_agents = PATHS.OUT_T / f"{AOI}_od_agents.csv"
    agents.to_csv(out_agents, index=False)
    log.info(f"Wrote {out_agents.name} (N={len(agents)})")


if __name__ == "__main__":
    main()


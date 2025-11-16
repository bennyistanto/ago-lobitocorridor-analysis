"""
Step 13 — Project Synergies Overlay (sites & clusters)

Purpose
-------
Quantify proximity-based 'synergies' between last-mile project sites and other
investments (Gov/WB/Other), and optionally between priority clusters (Step 11)
and those investments.

Two tidy tables:
  1) Site-level synergies: distance to nearest investment by source,
     plus counts of investments within radii (e.g., 5/10/30 km).
  2) Cluster-level synergies: same metrics computed around cluster centroids
     (no heavy polygonization required).

Key design choices
------------------
- Distances are geodesic (WGS84) using pyproj.Geod (robust for Angola scale).
- We treat any incoming geometry (Point/Line/Polygon) by a *representative point*:
  points → as-is; lines/polys → .representative_point() (always inside the feature).
- Radii are km (configurable here); counts are per source layer.
- If a layer is missing/empty, columns are included but filled with NaN/0 to stay schema-stable.
- We DO NOT de-duplicate overlapping project layers (a feature existing in multiple sources
  is counted in each respective source—this is usually intended).

Inputs
------
Required:
- PATHS.SITES                : Diversifica Mais (or main) sites, points (WGS84)

Optional investment layers (configure paths in config.py; any may be missing):
- PROJECTS_GOV               : government projects (point/line/polygon)
- PROJECTS_WB                : WB/GEMS projects (point/line/polygon)
- PROJECTS_OTH               : other partners/NGOs/private (point/line/polygon)

Optional cluster context:
- outputs/rasters/{AOI}_priority_clusters_1km.tif   (from Step 11)
- outputs/tables/{AOI}_priority_clusters.csv        (from Step 11; for centroids)

Other:
- PARAMS.TARGET_GRID         : only used to sanity-check CRS/extent if clustering is used

Outputs
-------
- outputs/tables/{AOI}_site_synergies.csv
- outputs/tables/{AOI}_cluster_synergies.csv        (only if Step 11 outputs exist)

Configuration knobs (local here; feel free to move into config.Params)
----------------------------------------------------------------------
- SYNERGY_RADII_KM = (5, 10, 30)

Usage
-----
In your notebook:

    import importlib
    m = importlib.import_module("step_13_synergies_overlay")
    importlib.reload(m)
    m.main()

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import Point
from pyproj import Geod

from config import (
    AOI, PATHS, PARAMS, get_logger, out_t,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------
# Radii from config (falls back to 5/10/30)
SYNERGY_RADII_KM: Tuple[int, ...] = tuple(getattr(PARAMS, "SYNERGY_RADII_KM", (5, 10, 30)))
SYNERGY_RADII_KM = tuple(sorted({int(r) for r in SYNERGY_RADII_KM if int(r) > 0}))
# ---------------------------------------------------------------------


# Optional project layers (will be pulled from config if available)
try:
    from config import PROJECTS_GOV
except Exception:
    PROJECTS_GOV = None
try:
    from config import PROJECTS_WB
except Exception:
    PROJECTS_WB = None
try:
    from config import PROJECTS_OTH
except Exception:
    PROJECTS_OTH = None


# ========================== helpers ==========================

_G = Geod(ellps="WGS84")

def _geodesic_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Vectorized geodesic distance (km) using WGS84 ellipsoid.
    Inputs can be scalars or arrays (numpy broadcasting applies).
    """
    # _G.inv returns (fwd_az, back_az, distance_m)
    _, _, dist_m = _G.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def _as_points_rep(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of points:
      - Points: keep as-is
      - Lines/Polygons: take representative_point() (guaranteed inside geometry)
    Keeps attributes except geometry is replaced by point.
    """
    if gdf.empty:
        return gdf
    # ensure WGS84
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    def _rep(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type == "Point":
            return geom
        return geom.representative_point()
    out = gdf.copy()
    out["geometry"] = out.geometry.apply(_rep)
    out = out[~out.geometry.isna()].copy()
    return out


def _load_sites() -> gpd.GeoDataFrame:
    """Load primary sites (PATHS.SITES). Returns points in EPSG:4326."""
    fp = Path(PATHS.SITES)
    if not fp.exists():
        raise FileNotFoundError(f"SITES not found: {fp}")
    g = gpd.read_file(fp)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    else:
        g = g.to_crs("EPSG:4326")
    g = g[~g.geometry.is_empty & g.geometry.notnull()].copy()
    if g.empty:
        log.warning("SITES layer is empty.")
    return g


def _load_optional_layer(fp: Optional[Path], tag: str) -> gpd.GeoDataFrame:
    """
    Load an optional investment layer (point/line/polygon) and convert to point representatives.
    Returns an empty (tagged) GeoDataFrame if missing/empty to keep logic consistent.
    """
    cols = ["source", "geometry"]
    if not fp:
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")
    p = Path(fp)
    if not p.exists():
        log.info(f"{tag} layer not found: {p.name}")
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")
    try:
        g = gpd.read_file(p)
        g = _as_points_rep(g)
        g["source"] = tag
        log.info(f"Loaded {len(g)} {tag} feature(s) from {p.name}")
        return g[cols]
    except Exception as e:
        log.warning(f"Failed reading {tag} layer {p}: {e}")
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")


def _nearest_and_counts(
    origins: gpd.GeoDataFrame,
    targets: gpd.GeoDataFrame,
    radii_km: Iterable[int],
) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    """
    For each origin point, compute:
      - nearest distance (km) to any target (np.nan if no targets)
      - count of targets within each radius (km)
    Returns:
      nearest_km: shape (N_origins,)
      counts: dict radius_km -> array (N_origins,)
    """
    n = len(origins)
    if targets.empty or n == 0:
        nearest = np.full(n, np.nan, dtype="float32")
        counts = {r: np.zeros(n, dtype="int32") for r in radii_km}
        return nearest, counts

    # Build arrays (lon/lat) for vectorized geodesic distance
    o_lon = origins.geometry.x.values
    o_lat = origins.geometry.y.values
    t_lon = targets.geometry.x.values
    t_lat = targets.geometry.y.values

    # nearest distance: compute in chunks to control memory if needed
    nearest = np.full(n, np.inf, dtype="float32")
    chunk = max(1, int(5000 / max(1, len(t_lon))))  # heuristic; small chunks if many targets
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        # distances matrix (i1-i0, M)
        D = _geodesic_km(o_lon[i0:i1, None], o_lat[i0:i1, None], t_lon[None, :], t_lat[None, :])
        nearest[i0:i1] = np.min(D, axis=1).astype("float32")

    # counts by radii
    counts = {}
    for r in radii_km:
        c = np.zeros(n, dtype="int32")
        # chunked again
        for i0 in range(0, n, chunk):
            i1 = min(n, i0 + chunk)
            D = _geodesic_km(o_lon[i0:i1, None], o_lat[i0:i1, None], t_lon[None, :], t_lat[None, :])
            c[i0:i1] = (D <= float(r)).sum(axis=1).astype("int32")
        counts[r] = c
    return nearest, counts


# ========================== main ==========================

def main() -> None:
    """
    Build site-level and (if available) cluster-level synergies tables:
      - dist to nearest Gov/WB/Oth investment (km)
      - counts of Gov/WB/Oth within 5/10/30 km
    Save to outputs/tables.
    """
    # Load sites
    sites = _load_sites()
    n_sites = len(sites)
    log.info(f"Loaded {n_sites} site(s).")

    # Load optional investment layers
    g_gov = _load_optional_layer(PROJECTS_GOV, "gov")
    g_wb  = _load_optional_layer(PROJECTS_WB,  "wb")
    g_oth = _load_optional_layer(PROJECTS_OTH, "oth")

    # --- Site-level synergies -------------------------------------------------
    # Nearest distances
    nearest_gov, counts_gov = _nearest_and_counts(sites, g_gov, SYNERGY_RADII_KM)
    nearest_wb,  counts_wb  = _nearest_and_counts(sites, g_wb,  SYNERGY_RADII_KM)
    nearest_oth, counts_oth = _nearest_and_counts(sites, g_oth, SYNERGY_RADII_KM)

    # Assemble DataFrame
    df_site = pd.DataFrame({
        "site_id": np.arange(1, n_sites + 1, dtype="int32"),
        "lon": sites.geometry.x.values.astype("float64"),
        "lat": sites.geometry.y.values.astype("float64"),
        "dist_km_nearest_gov": nearest_gov,
        "dist_km_nearest_wb":  nearest_wb,
        "dist_km_nearest_oth": nearest_oth,
    })
    for r in SYNERGY_RADII_KM:
        df_site[f"count_gov_le{r}km"] = counts_gov[r]
        df_site[f"count_wb_le{r}km"]  = counts_wb[r]
        df_site[f"count_oth_le{r}km"] = counts_oth[r]

    # --- schema lock for site synergies ---
    exp = [
        "site_id", "lon", "lat",
        "dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth",
    ]
    for r in SYNERGY_RADII_KM:
        exp += [f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"]

    for c in exp:
        if c not in df_site.columns:
            df_site[c] = np.nan

    # dtypes/rounding
    df_site["site_id"] = pd.to_numeric(df_site["site_id"], errors="coerce").round(0).astype("Int64")
    for c in ("lon", "lat"):
        df_site[c] = pd.to_numeric(df_site[c], errors="coerce").astype("float64").round(5)
    for c in ("dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth"):
        df_site[c] = pd.to_numeric(df_site[c], errors="coerce").astype("float64").round(1)
    for r in SYNERGY_RADII_KM:
        for c in (f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"):
            df_site[c] = pd.to_numeric(df_site[c], errors="coerce").fillna(0).round(0).astype("Int64")

    df_site = df_site[exp]

    out_csv_site = Path(out_t("site_synergies"))
    df_site.to_csv(out_csv_site, index=False)
    log.info(f"Saved site synergies → {out_csv_site.name} | rows={len(df_site)}")


    # --- Cluster-level synergies (optional) -----------------------------------
    clust_csv = PATHS.OUT_T / f"{AOI}_priority_clusters.csv"
    clust_tif = PATHS.OUT_R / f"{AOI}_priority_clusters_1km.tif"
    if clust_csv.exists() and clust_tif.exists():
        # Load cluster centroids (from Step 11 CSV)
        df_c = pd.read_csv(clust_csv)
        if {"cluster_id", "centroid_lon", "centroid_lat"}.issubset(df_c.columns) and len(df_c):
            # Make GeoDataFrame of centroids
            g_cent = gpd.GeoDataFrame(
                df_c[["cluster_id", "centroid_lon", "centroid_lat"]].copy(),
                geometry=[Point(xy) for xy in zip(df_c["centroid_lon"], df_c["centroid_lat"])],
                crs="EPSG:4326",
            )

            # Reuse nearest/counts
            cg, cg_counts = _nearest_and_counts(g_cent, g_gov, SYNERGY_RADII_KM)
            cw, cw_counts = _nearest_and_counts(g_cent, g_wb,  SYNERGY_RADII_KM)
            co, co_counts = _nearest_and_counts(g_cent, g_oth, SYNERGY_RADII_KM)

            df_cluster = pd.DataFrame({
                "cluster_id": df_c["cluster_id"].values.astype("int32"),
                "lon": df_c["centroid_lon"].values.astype("float64"),
                "lat": df_c["centroid_lat"].values.astype("float64"),
                "dist_km_nearest_gov": cg,
                "dist_km_nearest_wb":  cw,
                "dist_km_nearest_oth": co,
            })
            for r in SYNERGY_RADII_KM:
                df_cluster[f"count_gov_le{r}km"] = cg_counts[r]
                df_cluster[f"count_wb_le{r}km"]  = cw_counts[r]
                df_cluster[f"count_oth_le{r}km"] = co_counts[r]

            # --- schema lock for cluster synergies ---
            exp_c = [
                "cluster_id", "lon", "lat",
                "dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth",
            ]
            for r in SYNERGY_RADII_KM:
                exp_c += [f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"]

            for c in exp_c:
                if c not in df_cluster.columns:
                    df_cluster[c] = np.nan

            df_cluster["cluster_id"] = pd.to_numeric(df_cluster["cluster_id"], errors="coerce").round(0).astype("Int64")
            for c in ("lon", "lat"):
                df_cluster[c] = pd.to_numeric(df_cluster[c], errors="coerce").astype("float64").round(5)
            for c in ("dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth"):
                df_cluster[c] = pd.to_numeric(df_cluster[c], errors="coerce").astype("float64").round(1)
            for r in SYNERGY_RADII_KM:
                for c in (f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"):
                    df_cluster[c] = pd.to_numeric(df_cluster[c], errors="coerce").fillna(0).round(0).astype("Int64")

            df_cluster = df_cluster[exp_c]

            out_csv_cluster = Path(out_t("cluster_synergies"))
            df_cluster.to_csv(out_csv_cluster, index=False)
            log.info(f"Saved cluster synergies → {out_csv_cluster.name} | rows={len(df_cluster)}")

        else:
            log.info("Cluster CSV found but missing required columns; skipping cluster synergies.")
    else:
        log.info("Step 11 cluster outputs not found; skipping cluster synergies.")

    log.info("Step 13 complete.")


if __name__ == "__main__":
    main()

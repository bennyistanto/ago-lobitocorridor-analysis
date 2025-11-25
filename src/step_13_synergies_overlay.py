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
from pyproj import Geod, Transformer

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


_G = Geod(ellps="WGS84")


# ========================== helpers ==========================

def _xy_arrays(g: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) as strict 1D float64 arrays, same length, finite-only indices kept."""
    if g.empty:
        return np.array([], dtype="float64"), np.array([], dtype="float64")
    x = np.asarray(g.geometry.x, dtype="float64")
    y = np.asarray(g.geometry.y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _geodesic_km(lon1: np.ndarray, lat1: np.ndarray,
                 lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """
    Vectorized geodesic distance (km) using WGS84 ellipsoid.
    Inputs can be scalars or arrays (numpy broadcasting applies).
    """
    lon1 = np.asarray(lon1, dtype="float64")
    lat1 = np.asarray(lat1, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    _, _, dist_m = _G.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def _cluster_synergy_columns() -> list[str]:
    cols = [
        "cluster_id", "lon", "lat",
        "dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth",
    ]
    for r in SYNERGY_RADII_KM:
        cols += [f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"]
    return cols


def _write_empty_cluster_synergies(reason: str) -> None:
    out_csv_cluster = Path(out_t("cluster_synergies"))
    out_csv_cluster.parent.mkdir(parents=True, exist_ok=True)

    df_empty = pd.DataFrame(columns=_cluster_synergy_columns())
    df_empty.to_csv(out_csv_cluster, index=False)

    log.warning(
        "Step 13: writing EMPTY cluster_synergies table.\n"
        f"  Reason: {reason}\n"
        f"  Path  : {out_csv_cluster}"
    )


def _centroids_from_cluster_raster(clust_tif: Path) -> pd.DataFrame:
    """
    Compute centroid_lon/centroid_lat from the labeled cluster raster.
    Works even if the Step 11 CSV is empty or lacks centroid columns.
    """
    da = rxr.open_rasterio(clust_tif).squeeze(drop=True)
    arr = da.values
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        return pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])

    flat = arr.ravel()
    valid = np.isfinite(flat)

    nodata = da.rio.nodata
    if nodata is not None:
        valid &= (flat != nodata)

    valid &= (flat > 0)
    if valid.sum() == 0:
        return pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])

    ids = flat[valid].astype(np.int32)

    ny, nx = arr.shape
    pos = np.flatnonzero(valid)
    y_idx = pos // nx
    x_idx = pos % nx

    xcoords = np.asarray(da["x"].values, dtype="float64")
    ycoords = np.asarray(da["y"].values, dtype="float64")
    xs = xcoords[x_idx]
    ys = ycoords[y_idx]

    max_id = int(ids.max())
    counts = np.bincount(ids, minlength=max_id + 1).astype(np.float64)
    sumx = np.bincount(ids, weights=xs, minlength=max_id + 1)
    sumy = np.bincount(ids, weights=ys, minlength=max_id + 1)

    cid = np.nonzero(counts > 0)[0]
    cx = sumx[cid] / counts[cid]
    cy = sumy[cid] / counts[cid]

    # Convert to lon/lat if raster CRS is not EPSG:4326
    crs = da.rio.crs
    if crs is not None and str(crs).upper() not in ("EPSG:4326",):
        tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tr.transform(cx, cy)
    else:
        lon, lat = cx, cy

    return pd.DataFrame({"cluster_id": cid.astype(int), "centroid_lon": lon, "centroid_lat": lat})


def _as_points_rep(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of points:
      - Points: keep as-is
      - Lines/Polygons: take representative_point() (inside geometry)
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
    Load an optional investment layer (point/line/polygon) and convert to
    point representatives. Returns an empty (tagged) GeoDataFrame if missing
    or failed to read, to keep logic consistent.
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

    # sanitize coords → compact arrays for computation
    o_lon, o_lat = _xy_arrays(origins)
    t_lon, t_lat = _xy_arrays(targets)

    # Map back to original index positions (so outputs align with 'origins')
    orig_idx = np.nonzero(
        np.isfinite(np.asarray(origins.geometry.x)) &
        np.isfinite(np.asarray(origins.geometry.y))
    )[0]

    if o_lon.size == 0 or t_lon.size == 0 or orig_idx.size == 0:
        nearest = np.full(n, np.nan, dtype="float32")
        counts = {r: np.zeros(n, dtype="int32") for r in radii_km}
        return nearest, counts

    nearest = np.full(n, np.inf, dtype="float32")
    counts = {r: np.zeros(n, dtype="int32") for r in radii_km}

    chunk = max(1, int(5000 / max(1, t_lon.size)))

    # --- nearest distance ----------------------------------------------------
    for i0 in range(0, o_lon.size, chunk):
        i1 = min(o_lon.size, i0 + chunk)
        try:
            D = _geodesic_km(
                o_lon[i0:i1, None],
                o_lat[i0:i1, None],
                t_lon[None, :],
                t_lat[None, :],
            )
        except Exception:
            # Fallback: per-origin call
            D = np.empty((i1 - i0, t_lon.size), dtype="float64")
            for k, oi in enumerate(range(i0, i1)):
                lon1 = np.full(t_lon.size, o_lon[oi], dtype="float64")
                lat1 = np.full(t_lon.size, o_lat[oi], dtype="float64")
                _, _, dist_m = _G.inv(lon1, lat1, t_lon, t_lat)
                D[k, :] = dist_m / 1000.0

        nearest[orig_idx[i0:i1]] = np.min(D, axis=1).astype("float32")

    # --- counts within each radius ------------------------------------------
    for r in radii_km:
        c_compact = np.zeros(o_lon.size, dtype="int32")
        for i0 in range(0, o_lon.size, chunk):
            i1 = min(o_lon.size, i0 + chunk)
            try:
                D = _geodesic_km(
                    o_lon[i0:i1, None],
                    o_lat[i0:i1, None],
                    t_lon[None, :],
                    t_lat[None, :],
                )
            except Exception:
                D = np.empty((i1 - i0, t_lon.size), dtype="float64")
                for k, oi in enumerate(range(i0, i1)):
                    lon1 = np.full(t_lon.size, o_lon[oi], dtype="float64")
                    lat1 = np.full(t_lon.size, o_lat[oi], dtype="float64")
                    _, _, dist_m = _G.inv(lon1, lat1, t_lon, t_lat)
                    D[k, :] = dist_m / 1000.0

            c_compact[i0:i1] = (D <= float(r)).sum(axis=1).astype("int32")

        counts[r][orig_idx] = c_compact

    return nearest, counts


def _site_synergy_columns() -> list[str]:
    """Canonical ordering of columns for the site synergies table."""
    cols = [
        "site_id", "lon", "lat",
        "dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth",
    ]
    for r in SYNERGY_RADII_KM:
        cols += [f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"]
    return cols


def _write_empty_site_synergies(reason: str) -> None:
    """
    Write an EMPTY site synergies table with the expected columns and log the reason.

    Used when the site layer is missing or empty so that the 00–14 pipeline
    can continue without errors.
    """
    cols = _site_synergy_columns()
    df = pd.DataFrame(columns=cols)
    out_csv_site = Path(out_t("site_synergies"))
    out_csv_site.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_site, index=False)
    log.warning(
        "Step 13: writing EMPTY site_synergies table.\n"
        f"  Reason: {reason}\n"
        f"  Path  : {out_csv_site}"
    )


# ========================== main ==========================

def main() -> None:
    """
    Build site-level and (if available) cluster-level synergies tables:
      - dist to nearest Gov/WB/Oth investment (km)
      - counts of Gov/WB/Oth within configured radii
    Save to outputs/tables.
    """

    # ------------------------------------------------------------------
    # 1) Load sites, tolerating missing SITES layer
    # ------------------------------------------------------------------
    try:
        sites = _load_sites()
        n_sites = len(sites)
        log.info(f"Loaded {n_sites} site(s).")
    except FileNotFoundError as e:
        log.warning(
            "SITES layer missing for AOI=%s. Site-level synergies will be empty. (%s)",
            AOI, e
        )
        _write_empty_site_synergies("SITES shapefile not found for this AOI.")
        sites = None
        n_sites = 0

    # ------------------------------------------------------------------
    # 2) Load optional investment layers (used by both sites & clusters)
    # ------------------------------------------------------------------
    g_gov = _load_optional_layer(PROJECTS_GOV, "gov")
    g_wb  = _load_optional_layer(PROJECTS_WB,  "wb")
    g_oth = _load_optional_layer(PROJECTS_OTH, "oth")

    # ------------------------------------------------------------------
    # 3) Site-level synergies (skip cleanly if no sites)
    # ------------------------------------------------------------------
    if sites is not None and n_sites > 0:
        # Nearest distances
        nearest_gov, counts_gov = _nearest_and_counts(sites, g_gov, SYNERGY_RADII_KM)
        nearest_wb,  counts_wb  = _nearest_and_counts(sites, g_wb,  SYNERGY_RADII_KM)
        nearest_oth, counts_oth = _nearest_and_counts(sites, g_oth, SYNERGY_RADII_KM)

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

        # Schema lock
        exp = _site_synergy_columns()
        for c in exp:
            if c not in df_site.columns:
                df_site[c] = np.nan

        df_site["site_id"] = pd.to_numeric(df_site["site_id"], errors="coerce").round(0).astype("Int64")
        for c in ("lon", "lat"):
            df_site[c] = pd.to_numeric(df_site[c], errors="coerce").astype("float64").round(5)
        for c in ("dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth"):
            df_site[c] = pd.to_numeric(df_site[c], errors="coerce").astype("float64").round(1)
        for r in SYNERGY_RADII_KM:
            for c in (f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"):
                df_site[c] = (
                    pd.to_numeric(df_site[c], errors="coerce")
                    .fillna(0)
                    .round(0)
                    .astype("Int64")
                )

        df_site = df_site[exp]
        out_csv_site = Path(out_t("site_synergies"))
        out_csv_site.parent.mkdir(parents=True, exist_ok=True)
        df_site.to_csv(out_csv_site, index=False)
        log.info(f"Saved site synergies → {out_csv_site.name} | rows={len(df_site)}")

    else:
        # Already wrote empty table if SITES missing; if it's just empty, do it now.
        if sites is not None and n_sites == 0:
            _write_empty_site_synergies("SITES shapefile exists but contains zero usable sites.")
        log.info("No sites available → skipping site-level synergies.")

    # ------------------------------------------------------------------
    # 4) Cluster-level synergies (independent of whether sites exist)
    # ------------------------------------------------------------------
    clust_csv = PATHS.OUT_T / f"{AOI}_priority_clusters.csv"
    clust_tif = PATHS.OUT_R / f"{AOI}_priority_clusters_1km.tif"
    out_csv_cluster = Path(out_t("cluster_synergies"))
    out_csv_cluster.parent.mkdir(parents=True, exist_ok=True)

    # ---- helper (local) ------------------------------------------------
    def _clean_centroids(df: pd.DataFrame) -> pd.DataFrame:
        """Return df[cluster_id, centroid_lon, centroid_lat] with finite coords only."""
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])

        out = df.copy()
        out["centroid_lon"] = pd.to_numeric(out["centroid_lon"], errors="coerce")
        out["centroid_lat"] = pd.to_numeric(out["centroid_lat"], errors="coerce")
        out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce")

        out = out[np.isfinite(out["centroid_lon"]) & np.isfinite(out["centroid_lat"])].copy()
        out = out[np.isfinite(out["cluster_id"])].copy()
        if len(out) == 0:
            return pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])

        out["cluster_id"] = out["cluster_id"].round(0).astype("Int64")
        return out[["cluster_id", "centroid_lon", "centroid_lat"]]
    # -------------------------------------------------------------------

    # Strategy:
    # 1) Prefer centroids from Step 11 CSV if available.
    # 2) If CSV exists but no centroid columns (or empty), derive from the cluster raster.
    # 3) If no clusters exist, WRITE an empty cluster_synergies table (schema-stable) and continue.

    cent = pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])

    if clust_tif.exists():
        # Try from CSV first
        if clust_csv.exists():
            try:
                df_c = pd.read_csv(clust_csv)
            except Exception as e:
                log.warning(f"Failed reading cluster CSV {clust_csv.name}: {e}")
                df_c = pd.DataFrame()

            if {"cluster_id", "centroid_lon", "centroid_lat"}.issubset(df_c.columns) and len(df_c):
                cent = _clean_centroids(df_c[["cluster_id", "centroid_lon", "centroid_lat"]])
                if len(cent) == 0:
                    log.info("Cluster CSV had centroid columns but no valid (finite) centroid rows; will derive from raster.")
            else:
                log.info("Cluster CSV found but missing centroid columns (or empty); will derive centroids from raster.")
        else:
            log.info("Cluster CSV not found; will derive centroids from raster.")

        # Derive from raster if needed
        if len(cent) == 0:
            try:
                cent = _centroids_from_cluster_raster(clust_tif)
                cent = _clean_centroids(cent)
            except Exception as e:
                log.warning(f"Failed deriving centroids from cluster raster {clust_tif.name}: {e}")
                cent = pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])
    else:
        # No raster -> can't derive. Write empty file so downstream tables don't crash.
        _write_empty_cluster_synergies("Missing Step 11 cluster raster.")
        log.info("Step 11 cluster outputs not found (cluster raster missing). Wrote empty cluster synergies.")
        cent = pd.DataFrame(columns=["cluster_id", "centroid_lon", "centroid_lat"])

    # If still no centroids, it generally means: 0 clusters (connected components = 0)
    if len(cent) == 0:
        _write_empty_cluster_synergies("No clusters present (0 connected components) or no valid centroids.")
        log.info("No clusters available → wrote empty cluster synergies.")
    else:
        # Build GeoDataFrame of centroids (WGS84)
        g_cent = gpd.GeoDataFrame(
            cent[["cluster_id"]].copy(),
            geometry=[Point(xy) for xy in zip(cent["centroid_lon"], cent["centroid_lat"])],
            crs="EPSG:4326",
        )
        g_cent = g_cent[~g_cent.geometry.is_empty & g_cent.geometry.notnull()].copy()

        cg, cg_counts = _nearest_and_counts(g_cent, g_gov, SYNERGY_RADII_KM)
        cw, cw_counts = _nearest_and_counts(g_cent, g_wb,  SYNERGY_RADII_KM)
        co, co_counts = _nearest_and_counts(g_cent, g_oth, SYNERGY_RADII_KM)

        df_cluster = pd.DataFrame({
            "cluster_id": cent["cluster_id"].astype("Int64"),
            "lon": cent["centroid_lon"].astype("float64"),
            "lat": cent["centroid_lat"].astype("float64"),
            "dist_km_nearest_gov": cg,
            "dist_km_nearest_wb":  cw,
            "dist_km_nearest_oth": co,
        })

        for r in SYNERGY_RADII_KM:
            df_cluster[f"count_gov_le{r}km"] = cg_counts[r]
            df_cluster[f"count_wb_le{r}km"]  = cw_counts[r]
            df_cluster[f"count_oth_le{r}km"] = co_counts[r]

        # Schema lock
        exp_c = _cluster_synergy_columns()
        for c in exp_c:
            if c not in df_cluster.columns:
                df_cluster[c] = np.nan

        # Types/rounding
        df_cluster["cluster_id"] = pd.to_numeric(df_cluster["cluster_id"], errors="coerce").round(0).astype("Int64")
        for c in ("lon", "lat"):
            df_cluster[c] = pd.to_numeric(df_cluster[c], errors="coerce").astype("float64").round(5)
        for c in ("dist_km_nearest_gov", "dist_km_nearest_wb", "dist_km_nearest_oth"):
            df_cluster[c] = pd.to_numeric(df_cluster[c], errors="coerce").astype("float64").round(1)
        for r in SYNERGY_RADII_KM:
            for c in (f"count_gov_le{r}km", f"count_wb_le{r}km", f"count_oth_le{r}km"):
                df_cluster[c] = (
                    pd.to_numeric(df_cluster[c], errors="coerce")
                    .fillna(0)
                    .round(0)
                    .astype("Int64")
                )

        df_cluster = df_cluster[exp_c]
        df_cluster.to_csv(out_csv_cluster, index=False)
        log.info(f"Saved cluster synergies → {out_csv_cluster.name} | rows={len(df_cluster)}")

    log.info("Step 13 complete.")


if __name__ == "__main__":
    main()

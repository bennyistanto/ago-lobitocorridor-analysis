"""
Step 14 — OD-Lite v2
--------------------

Admin2-based gravity model to sketch origin–destination flows for corridor
storytelling and screening (not planning-grade). Keeps the “lite” structure,
but adds:
  • Selectable impedance function f(D): exponential or power.
  • Optional doubly-constrained balancing (IPF) to match productions/attractions.
  • Cutoff by maximum distance (km).
  • Equity tilt using RWI (Meta, -2..+2) as mass modifier (optional).
  • Diagnostics & basic QA of the flow matrix.

Inputs (canonical):
  - OUT_R:  <AOI>_pop_1km.tif        (Step 00)
  - OUT_R:  rwi_meta_1km.tif         (Step 00, optional)
  - MUNI:   Admin2 shapefile (poverty/pop/government themes; first existing)

Outputs (tables):
  - OUT_T:  <AOI>_od_gravity.csv     (oi,dj,flow,dist_km)
  - OUT_T:  <AOI>_od_zone_attrs.csv  (zone attributes + centroids)
  - OUT_T:  <AOI>_od_agents.csv      (sampled agents for quick viz)

Notes
-----
• This is a *screening* tool. If you need planning-grade OD:
  - Replace geodesic centroid–centroid distance with a skim matrix of travel times.
  - Calibrate impedance parameters (lambda/beta) to observed flows.
  - Use proper zoning, productions, and attractions by activity type.

"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from shapely.geometry import Point

from config import (
    AOI, PATHS, PARAMS, get_logger,
    out_t, out_r, muni_path_for, ADMIN2_THEMES,
    RESAMPLE,
)

from utils_geo import (
    open_template, align_to_template
)

log = get_logger(__name__)

# -------------------------------
# Parameters (with safe defaults)
# -------------------------------
# Gravity parameters (can be set in PARAMS to override defaults)
ALPHA   = float(getattr(PARAMS, "OD_ALPHA", 1.0))       # exponent on origin mass
GAMMA   = float(getattr(PARAMS, "OD_GAMMA", 1.0))       # exponent on destination mass

# Choose impedance: "exp" uses exp(-lambda*D), "pow" uses (1 + D)**(-beta)
F_TYPE  = str(getattr(PARAMS, "OD_F", "exp")).lower()
LAMBDA  = float(getattr(PARAMS, "OD_LAMBDA", 0.015))    # used if F_TYPE == "exp"
BETA    = float(getattr(PARAMS, "OD_BETA", 1.5))        # used if F_TYPE == "pow"

# Total trips to scale to (screening target)
TRIPS_TOTAL = float(getattr(PARAMS, "OD_TRIPS_TOTAL", 1_000_000.0))

# Max distance cutoff (km) — set None or very large to disable
MAX_DIST_KM = getattr(PARAMS, "OD_MAX_DIST_KM", 1500.0)
MAX_DIST_KM = float(MAX_DIST_KM) if MAX_DIST_KM is not None else None

# Use doubly-constrained gravity (IPF) to match row/col totals
USE_DOUBLY_CONSTRAINED = bool(getattr(PARAMS, "OD_USE_DOUBLY_CONSTRAINED", False))

# Mass tilt using RWI (poverty proxy): enabled + weight
USE_RWI_IN_MASS = bool(getattr(PARAMS, "USE_RWI_IN_MASS", True))
RWI_WEIGHT      = float(getattr(PARAMS, "RWI_WEIGHT", 0.25))  # how much to tilt mass

# How many agents to sample from F for a quick viz layer
N_AGENTS = int(getattr(PARAMS, "OD_N_AGENTS", 50_000))

# Only Admin2 zones are implemented for now
USE_ADMIN2_ZONES = True


# -----------------------
# Utility / helper funcs
# -----------------------
def _pick_admin2_geom() -> Optional[Path]:
    """
    Pick an Admin2 shapefile for this AOI.
    Preference order: poverty theme → other themes.
    `muni_path_for` returns list[Path]; return the first that exists.
    """
    cand = muni_path_for(AOI, "poverty")
    for p in cand:
        if p.exists():
            return p
    for theme in ADMIN2_THEMES:
        for p in muni_path_for(AOI, theme):
            if p.exists():
                return p
    return None


def _aggregate_pop_by_admin2(pop: xr.DataArray, g: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Zonal sum of population by Admin2 polygons.
    Returns a DataFrame indexed by Admin2 code with 'pop' column.
    """
    # Compute per-pixel area weights? Not needed for population (already absolute counts per cell)
    # Strategy: rasterize polygons to label raster is overkill here — sampling masks per polygon is fine.
    # Use bounding box window for speed (vectorized mask via rioxarray.clip seems OK for small N).
    rows = []
    # Normalize ID fields
    need = {"ADM2CD_c", "NAM_1", "NAM_2"}
    if not need.issubset(g.columns):
        raise ValueError(f"Admin2 file missing required columns {need}")

    for i, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Clip pop to polygon (mask=True -> outside set to NaN)
        try:
            sub = pop.rio.clip([geom.__geo_interface__], pop.rio.crs, drop=True, invert=False)
        except Exception as e:
            log.warning(f"Clip failed for Admin2={row.get('ADM2CD_c')}: {e}")
            continue
        val = float(np.nansum(sub.values))
        rows.append({
            "ADM2CD_c": row["ADM2CD_c"],
            "NAM_1": row["NAM_1"],
            "NAM_2": row["NAM_2"],
            "pop": val
        })
    df = pd.DataFrame(rows).set_index("ADM2CD_c").sort_index()
    # Guard: zero pop zones could exist — retain but warn
    if (df["pop"] <= 0).any():
        z = int((df["pop"] <= 0).sum())
        log.warning(f"{z} Admin2 zone(s) have zero/negative population (kept).")
    return df


def _merge_rwi_as_mass_tilt(df: pd.DataFrame, rwi: xr.DataArray, g: gpd.GeoDataFrame, weight: float = 0.25) -> pd.DataFrame:
    """
    Merge RWI (Meta -2..+2) as a mass tilt factor for equity:
      pop_tilt = pop * (1 + weight * norm_rwi)
    where norm_rwi is scaled to [-1, +1].
    """
    # Average RWI within each zone
    vals = []
    for i, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            vals.append(np.nan)
            continue
        try:
            sub = rwi.rio.clip([geom.__geo_interface__], rwi.rio.crs, drop=True, invert=False)
        except Exception:
            vals.append(np.nan)
            continue
        vals.append(np.nanmean(sub.values))
    s = pd.Series(vals, index=g["ADM2CD_c"].values, name="rwi_mean_zone")
    # Normalize [-1,1] from [-2,2]
    s_norm = s / 2.0
    df2 = df.join(s_norm.rename("rwi_z"), how="left")
    df2["mass"] = df2["pop"] * (1.0 + weight * df2["rwi_z"].fillna(0.0))
    # non-negativity guard
    df2.loc[df2["mass"] < 0, "mass"] = 0.0
    return df2


def _pairwise_geodesic_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Pairwise geodesic distance (km) between lon/lat arrays using haversine.
    Returns an (N,N) symmetric matrix with zeros on the diagonal.
    """
    # radians
    lonr = np.radians(lon).reshape(-1, 1)
    latr = np.radians(lat).reshape(-1, 1)
    lonrT = lonr.T
    latrT = latr.T
    dlon = lonr - lonrT
    dlat = latr - latrT
    a = np.sin(dlat / 2.0) ** 2 + np.cos(latr) * np.cos(latrT) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return 6371.0088 * c  # Earth's mean radius km


def _impedance(D_km: np.ndarray, f_type: str, lam: float, beta: float) -> np.ndarray:
    """
    Compute impedance f(D) given a matrix of distances in km.
    f_type: "exp"  -> exp(-lambda * D)
            "pow"  -> (1 + D)^(-beta)
    """
    if f_type == "exp":
        return np.exp(-lam * D_km)
    elif f_type == "pow":
        return np.power(1.0 + D_km, -beta)
    else:
        raise ValueError(f"Unknown OD_F='{f_type}'. Use 'exp' or 'pow'.")


def _apply_cutoff(D_km: np.ndarray, F0: np.ndarray, max_km: Optional[float]) -> np.ndarray:
    """Zero out flows beyond a maximum distance (km)."""
    if (max_km is None) or (not np.isfinite(max_km)):
        return F0
    mask = (D_km > float(max_km))
    F0 = F0.copy()
    F0[mask] = 0.0
    return F0


def _ipf_balance(F: np.ndarray, row_targets: np.ndarray, col_targets: np.ndarray, max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    """
    Simple Iterative Proportional Fitting (IPF) to match row/column totals.
    Any rows/cols with zero targets are handled gracefully.
    """
    F = F.copy()
    row_targets = row_targets.astype(float)
    col_targets = col_targets.astype(float)
    # Avoid division by zero
    row_targets = np.where(row_targets < 1e-12, 0.0, row_targets)
    col_targets = np.where(col_targets < 1e-12, 0.0, col_targets)

    for it in range(max_iter):
        # Row scale
        rsum = F.sum(axis=1)
        scale_r = np.ones_like(rsum)
        nz = rsum > 0
        scale_r[nz] = np.divide(row_targets[nz], rsum[nz], out=np.ones_like(rsum[nz]), where=(rsum[nz] > 0))
        F *= scale_r[:, None]

        # Col scale
        csum = F.sum(axis=0)
        scale_c = np.ones_like(csum)
        nz = csum > 0
        scale_c[nz] = np.divide(col_targets[nz], csum[nz], out=np.ones_like(csum[nz]), where=(csum[nz] > 0))
        F *= scale_c[None, :]

        # Convergence check
        err_r = np.nanmax(np.abs(F.sum(axis=1) - row_targets)) if row_targets.sum() > 0 else 0.0
        err_c = np.nanmax(np.abs(F.sum(axis=0) - col_targets)) if col_targets.sum() > 0 else 0.0
        if max(err_r, err_c) < tol:
            log.info(f"IPF converged in {it+1} iterations (tol={tol}).")
            break
    else:
        log.warning(f"IPF did not converge within {max_iter} iterations (max err={max(err_r, err_c):.4g}).")
    return F


def _gravity_matrix(
    pop_i: np.ndarray,
    pop_j: np.ndarray,
    D_km: np.ndarray,
    alpha: float, gamma: float,
    f_type: str, lam: float, beta: float,
    trips_total: float,
    use_doubly_constrained: bool = False,
    max_dist_km: Optional[float] = None,
) -> np.ndarray:
    """
    Compute gravity matrix F_{ij} ∝ (pop_i^alpha)(pop_j^gamma) f(D_ij),
    scaled to trips_total. If doubly-constrained, IPF to match row/col totals
    proportional to pop_i and pop_j.
    """
    # Base potential (unscaled)
    Mi = np.power(np.maximum(pop_i, 0.0), alpha)
    Mj = np.power(np.maximum(pop_j, 0.0), gamma)
    imped = _impedance(D_km, f_type=f_type, lam=lam, beta=beta)

    # Raw flows
    F0 = (Mi[:, None] * Mj[None, :]) * imped

    # Apply cutoff
    F0 = _apply_cutoff(D_km, F0, max_dist_km)

    if F0.sum() <= 0:
        raise ValueError("Gravity produced zero total flow (check parameters and cutoff).")

    # Scale to trips_total
    F = (F0 / F0.sum()) * float(trips_total)

    if use_doubly_constrained:
        # Targets proportional to pop masses
        row_t = (pop_i / np.sum(pop_i)) * trips_total
        col_t = (pop_j / np.sum(pop_j)) * trips_total
        F = _ipf_balance(F, row_t, col_t)

    return F


def _sample_agents_from_flows(
    F: np.ndarray,
    lon: np.ndarray, lat: np.ndarray,
    n_agents: int = 50_000, seed: int = 42
) -> pd.DataFrame:
    """
    Multinomially sample OD pairs from normalized F to get agent samples.
    """
    rng = np.random.default_rng(seed)
    P = F / F.sum()
    flat = P.ravel()
    idx = rng.choice(flat.size, size=min(n_agents, flat.size), replace=True, p=flat)
    n = lon.size
    oi = (idx // n).astype(int)
    dj = (idx %  n).astype(int)
    return pd.DataFrame({
        "oi": oi, "dj": dj,
        "o_lon": lon[oi], "o_lat": lat[oi],
        "d_lon": lon[dj], "d_lat": lat[dj],
    })


# -------------
# Main routine
# -------------
def main() -> None:
    """
    Build an Admin2-based gravity OD matrix and sample agents.
    Writes three CSVs under outputs/tables for downstream viz.
    """
    if not USE_ADMIN2_ZONES:
        log.error("Only Admin2 zones are implemented in Step 14 for now. Set USE_ADMIN2_ZONES=True.")
        return

    # 1) Template & population raster
    T = open_template(PARAMS.TARGET_GRID)

    pop_fp = PATHS.OUT_R / f"{AOI}_pop_1km.tif"
    if not pop_fp.exists():
        raise FileNotFoundError(f"Population raster not found: {pop_fp.name}")

    pop = rxr.open_rasterio(pop_fp, masked=True).squeeze()
    if (pop.shape != T.shape) or (pop.rio.transform() != T.rio.transform()) or (pop.rio.crs != T.rio.crs):
        pop = pop.rio.reproject_match(T, resampling=RESAMPLE("bilinear"))

    # Optional RWI (Meta -2..+2), aligned to template
    rwi_fp = out_r("rwi_meta_1km")
    rwi = None
    if Path(rwi_fp).exists():
        rwi = rxr.open_rasterio(rwi_fp, masked=True).squeeze()
        if (rwi.shape != T.shape) or (rwi.rio.transform() != T.rio.transform()) or (rwi.rio.crs != T.rio.crs):
            rwi = rwi.rio.reproject_match(T, resampling=RESAMPLE("bilinear"))
        log.info("RWI found: using equity tilt.")

    # 2) Admin2 geometry
    adm2_fp = _pick_admin2_geom()
    if adm2_fp is None:
        raise FileNotFoundError("No Admin2 RAPP shapefile found for this AOI in PATHS.MUNI_DIR.")
    g = gpd.read_file(adm2_fp)
    g = g.to_crs("EPSG:4326") if g.crs else g.set_crs("EPSG:4326")
    g = g[~g.geometry.is_empty & g.geometry.notnull()].copy()

    need = {"ADM2CD_c", "NAM_1", "NAM_2"}
    if not need.issubset(g.columns):
        raise ValueError(f"{adm2_fp.name} missing required columns {need}")

    # 3) Zone masses
    df_z = _aggregate_pop_by_admin2(pop, g)          # index=ADM2CD_c, 'pop'
    if USE_RWI_IN_MASS and (rwi is not None):
        df_z = _merge_rwi_as_mass_tilt(df_z, rwi, g, weight=RWI_WEIGHT)
        mass_col = "mass"
    else:
        df_z["mass"] = df_z["pop"].clip(lower=0.0)
        mass_col = "mass"

    # 4) Zone centroids & pairwise distance
    cent = g[["ADM2CD_c", "NAM_1", "NAM_2", "geometry"]].copy()
    cent["centroid"] = cent.geometry.centroid
    cent["lon"] = cent["centroid"].x.astype("float64")
    cent["lat"] = cent["centroid"].y.astype("float64")

    # --- Align centroid order to df_z.index (critical for F↔D consistency)
    cent = cent.set_index("ADM2CD_c").loc[df_z.index].reset_index()

    D = _pairwise_geodesic_km(cent["lon"].values, cent["lat"].values)
    log.info("OD zones aligned by ADM2CD_c order | N=%d", len(cent))

    # 5) Gravity model
    pop_i = df_z[mass_col].values.copy()
    pop_j = df_z[mass_col].values.copy()

    F = _gravity_matrix(
        pop_i=pop_i,
        pop_j=pop_j,
        D_km=D,
        alpha=ALPHA, gamma=GAMMA,
        f_type=F_TYPE, lam=LAMBDA, beta=BETA,
        trips_total=TRIPS_TOTAL,
        use_doubly_constrained=USE_DOUBLY_CONSTRAINED,
        max_dist_km=MAX_DIST_KM,
    )

    # 6) Diagnostics
    # -- sanity checks
    total = F.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Flow matrix invalid (sum <= 0).")

    # -- distance-weighted stats
    #   (flatten while excluding diagonal)
    n = D.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)
    flows = F[mask_offdiag]
    dists = D[mask_offdiag]
    mean_d = float(np.average(dists, weights=flows)) if flows.sum() > 0 else float(np.nan)
    med_d = float(np.quantile(np.repeat(dists, np.maximum(flows.astype(int), 0)), 0.5)) if flows.sum() > 0 else float("nan")
    within_cut = None
    if MAX_DIST_KM is not None and np.isfinite(MAX_DIST_KM):
        within_cut = float(100.0 * flows[dists <= MAX_DIST_KM].sum() / flows.sum()) if flows.sum() > 0 else float("nan")

    log.info(f"Flows: total={total:,.0f} | mean_dist_km={mean_d:,.1f} | within_cutoff%={within_cut if within_cut is not None else 'NA'}")

    # 7) Save outputs
    out_grav = PATHS.OUT_T / f"{AOI}_od_gravity.csv"
    out_z    = PATHS.OUT_T / f"{AOI}_od_zone_attrs.csv"

    # Trip table (long)
    oi_idx = np.repeat(df_z.index.values, len(df_z))
    dj_idx = np.tile(df_z.index.values, len(df_z))
    pd.DataFrame({
        "oi": oi_idx,
        "dj": dj_idx,
        "flow": F.ravel().astype(float),
        "dist_km": D.ravel().astype(float),
    }).to_csv(out_grav, index=False)

    # Zone attributes (with centroids)
    # Reset index so ADM2CD_c becomes a column in the CSV
    df_z_out = (
        df_z.copy()
        .reset_index()                       # brings ADM2CD_c into columns
        .assign(
            NAM_1 = cent["NAM_1"].values,
            NAM_2 = cent["NAM_2"].values,
            lon   = cent["lon"].values,
            lat   = cent["lat"].values,
        )
    )

    # Stable, explicit column order (IDs first)
    cols = [c for c in ["ADM2CD_c", "NAM_1", "NAM_2", "lon", "lat", "pop", "rwi_z", "mass"] if c in df_z_out.columns]
    df_z_out = df_z_out[cols]
    df_z_out.to_csv(out_z, index=False)

    log.info(f"Wrote {out_grav.name}, {out_z.name}")

    # 8) Sample agents from F for quick viz
    agents = _sample_agents_from_flows(F, cent["lon"].values, cent["lat"].values, n_agents=N_AGENTS, seed=42)
    out_agents = PATHS.OUT_T / f"{AOI}_od_agents.csv"
    agents.to_csv(out_agents, index=False)
    log.info(f"Wrote {out_agents.name} (N={len(agents)})")

    # 9) OD-bottleneck overlay: rank flood-risk road cells by flow load
    _od_bottleneck_overlay(F, cent, D, df_z)

    log.info("Step 14 complete.")


def _od_bottleneck_overlay(
    F: np.ndarray,
    cent: pd.DataFrame,
    D: np.ndarray,
    df_z: pd.DataFrame,
) -> None:
    """
    Link OD flows to flood-risk road cells (Step 04 output).

    Approach (fast heuristic, no full routing)
    -------------------------------------------
    For each OD pair (i→j) with flow > 0, draw the straight-line segment between
    zone centroids and identify which 1-km grid cells are intercepted. Accumulate
    total flow onto each cell. Then intersect with the road-risk mask to rank
    bottleneck cells by total OD flow they would block if disrupted.

    Outputs
    -------
    - {AOI}_od_bottleneck_cells.csv  (x, y, flow_load, is_risk, risk_rank)
    - {AOI}_od_bottleneck_risk.tif   (flow_load on risk cells only)
    """
    from rasterio.transform import rowcol

    # Check for required inputs
    risk_fp = PATHS.OUT_R / f"{AOI}_roads_flood_risk_cells_1km.tif"
    if not Path(risk_fp).exists():
        log.warning("Road-risk raster not found (%s); skipping OD-bottleneck overlay.", risk_fp.name)
        return

    T = open_template(PARAMS.TARGET_GRID)
    risk = rxr.open_rasterio(risk_fp, masked=True).squeeze()
    if (risk.shape != T.shape) or (risk.rio.transform() != T.rio.transform()):
        risk = risk.rio.reproject_match(T, resampling=RESAMPLE("nearest"))

    tf = T.rio.transform()
    ny, nx = T.shape

    # Accumulate flow per grid cell using Bresenham-like traversal of OD segments
    flow_grid = np.zeros((ny, nx), dtype=np.float64)
    n_zones = len(cent)
    lons = cent["lon"].values
    lats = cent["lat"].values

    # Only process significant flows (top 90% of total flow for speed)
    flat_F = F.ravel()
    flat_D = D.ravel()
    oi_all = np.repeat(np.arange(n_zones), n_zones)
    dj_all = np.tile(np.arange(n_zones), n_zones)
    mask = (flat_F > 0) & (oi_all != dj_all)
    oi_sel = oi_all[mask]
    dj_sel = dj_all[mask]
    fl_sel = flat_F[mask]

    # Sort by flow descending; keep enough to cover 90% of total flow
    order = np.argsort(-fl_sel)
    cumflow = np.cumsum(fl_sel[order])
    total = cumflow[-1] if len(cumflow) > 0 else 0
    if total <= 0:
        log.warning("No OD flows to overlay; skipping bottleneck analysis.")
        return
    cutoff_idx = int(np.searchsorted(cumflow, 0.90 * total)) + 1
    order = order[:cutoff_idx]

    log.info("OD-bottleneck: processing %d OD pairs (90%% of flow)...", len(order))

    for idx in order:
        oi = oi_sel[idx]
        dj = dj_sel[idx]
        flow = fl_sel[idx]

        # Rasterize the line from centroid_i to centroid_j (Bresenham)
        r0, c0 = rowcol(tf, lons[oi], lats[oi])
        r1, c1 = rowcol(tf, lons[dj], lats[dj])

        # Clip to grid bounds
        if not (0 <= r0 < ny and 0 <= c0 < nx and 0 <= r1 < ny and 0 <= c1 < nx):
            continue

        # Simple DDA line rasterization
        cells = _bresenham(r0, c0, r1, c1, ny, nx)
        for r, c in cells:
            flow_grid[r, c] += flow

    # Intersect with risk mask
    risk_mask = np.nan_to_num(risk.values, nan=0.0) > 0.5
    risk_flow = np.where(risk_mask, flow_grid, 0.0)

    # Build output table of risk cells with their flow load
    rr, cc = np.where(risk_mask & (flow_grid > 0))
    if rr.size == 0:
        log.info("No OD flow intersects risk cells; bottleneck table empty.")
        return

    rows = []
    for i in range(len(rr)):
        lon = float(T.x.values[cc[i]])
        lat = float(T.y.values[rr[i]])
        rows.append({
            "row": int(rr[i]), "col": int(cc[i]),
            "lon": lon, "lat": lat,
            "flow_load": float(flow_grid[rr[i], cc[i]]),
        })

    btl = pd.DataFrame(rows).sort_values("flow_load", ascending=False).reset_index(drop=True)
    btl["risk_rank"] = range(1, len(btl) + 1)

    out_csv = out_t("od_bottleneck_cells")
    btl.to_csv(out_csv, index=False)
    log.info("Wrote OD-bottleneck table -> %s (%d risk cells with flow)", out_csv.name, len(btl))

    # Write raster (flow load on risk cells only)
    from utils_geo import write_gtiff
    risk_flow_da = xr.DataArray(risk_flow.astype("float32"), coords=T.coords, dims=T.dims)
    out_tif = out_r("od_bottleneck_risk")
    write_gtiff(risk_flow_da, out_tif, like=T, nodata=0)
    log.info("Wrote %s", Path(out_tif).name)


def _bresenham(r0: int, c0: int, r1: int, c1: int, ny: int, nx: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm returning list of (row, col) within grid bounds."""
    cells = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if 0 <= r < ny and 0 <= c < nx:
            cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    return cells


if __name__ == "__main__":
    main()

"""
Step 02 — Isochrone KPIs (population, cropland, electrification).

Reads (AOI-prefixed, produced by Step 00/01):
  - {AOI}_pop_1km.tif                  (persons per 1-km cell; already aligned)
  - {AOI}_cropland_fraction_1km.tif    (0..1 fraction of cropland per 1-km cell)
  - {AOI}_elec_grid_1km.tif            (1 if grid-present cell; NaN/0 otherwise)
  - {AOI}_iso_le_{thr}min_1km.tif      (1 inside isochrone, 0 outside, NaN masked)

Writes:
  - outputs/tables/{AOI}_kpis_isochrones.csv

Notes on denominators:
  - pop_pct           = pop within isochrone / total pop in AOI
  - cropland_pct      = cropland km² within isochrone / total cropland km² in AOI
  - electrified_pct   = electrified cells within isochrone / total electrified cells in AOI
    (i.e., *share of all electrified cells* captured by the isochrone, not area share)
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

from config import (
    AOI, PARAMS, out_r,  
    get_logger, KPI_CSV
)
from utils_geo import (
    open_template, zonal_sum, estimate_cell_area_km2
)

log = get_logger(__name__)


def _assert_same_shape(*das) -> None:
    """Raise AssertionError if any input rasters have differing shapes."""
    shapes = {da.shape for da in das}
    assert len(shapes) == 1, f"Rasters must share identical shape; got {shapes}"


# --- helper: monotonicity check (tiny float tolerance) -----------------------
def _assert_monotone_increasing(values, label: str) -> None:
    """Raise if sequence is not non-decreasing; tolerates tiny float jitter."""
    import numpy as _np
    arr = _np.asarray(values, dtype=float)
    diffs = _np.diff(arr)
    if (diffs < -1e-6).any():
        raise AssertionError(f"{label} is not monotone non-decreasing (check isochrone masks / nodata).")


def main() -> None:
    # -------------------------------------------------------------------------
    # Load aligned rasters (all should be on the 1-km target grid)
    # -------------------------------------------------------------------------
    pop     = open_template(out_r("pop_1km"))
    cl_frac = open_template(out_r("cropland_fraction_1km"))   # fractional 0..1
    elg     = open_template(out_r("elec_grid_1km"))

    # After loading `elg`
    uniq, counts = np.unique(elg.values[~np.isnan(elg.values)], return_counts=True)
    log.info("Electrification grid value counts (non-NaN): %s", dict(zip(uniq.astype(int), counts.tolist())))
    unexpected = [v for v in uniq if v not in (0, 1, 99)]
    if unexpected:
        log.warning("Unexpected codes in elec_grid_1km: %s (only 1=grid, 99/0/NaN expected).", unexpected)

    _assert_same_shape(pop, cl_frac, elg)

    # Log grid context once (handy when switching AOIs)
    tf = pop.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"KPI base rasters loaded | CRS={pop.rio.crs} | "
        f"size={pop.rio.height}x{pop.rio.width} | cell={resx:.4f}x{resy:.4f}"
    )
    log.info("Electrification mapping: FinalElecCode2020 -> elec_grid_1km == 1 (grid); 99/0/NaN = non-grid.")


    # -------------------------------------------------------------------------
    # Denominators for % computations  (area-true cropland; robust to EPSG:4326)
    # -------------------------------------------------------------------------
    pop_total = float(np.nansum(pop.values))

    # estimate_cell_area_km2 can return a scalar (≈1 km²) or a per-cell 2D array
    cell_km2_est = estimate_cell_area_km2(pop)

    # total cropland km² (area-true)
    if hasattr(cell_km2_est, "shape"):
        # per-cell area available ⇒ multiply element-wise
        cl_km2_tot = float(np.nansum(cl_frac.values * cell_km2_est))
        cell_km2_log = float(np.nanmean(cell_km2_est))  # for logging only
    else:
        # scalar fallback
        cl_km2_tot = float(np.nansum(cl_frac.values)) * float(cell_km2_est)
        cell_km2_log = float(cell_km2_est)

    # electrified cells (count of 1’s)
    elg_total = float(np.nansum(elg.values == 1))

    log.info(
        f"Denominators | pop_total={pop_total:,.0f} | "
        f"cropland_total_km2={cl_km2_tot:,.2f} | electrified_cells_total={elg_total:,.0f} | "
        f"cell_area_km2≈{cell_km2_log:.3f}"
    )


    # -------------------------------------------------------------------------
    # Loop thresholds (de-duplicate + sort for stable output)
    # -------------------------------------------------------------------------
    if not PARAMS.ISO_THRESH:
        log.warning("No ISO_THRESH provided; nothing to compute.")
        return

    rows: list[Dict[str, Any]] = []

    for thr in sorted(set(PARAMS.ISO_THRESH)):
        mask = open_template(out_r(f"iso_le_{thr}min_1km"))
        _assert_same_shape(mask, pop)

        # Robust boolean "inside" mask
        mv = mask.values
        # treat NaN as outside; allow float/bool/byte variants
        inside = np.where(np.isnan(mv), False, mv >= 0.5)

        # Precompute this once for counts
        cells_within = float(np.count_nonzero(inside))

        # Population within (zonal sum)
        # If your `zonal_sum` expects mask==1, keep it; else compute directly:
        pop_sum = float(zonal_sum(mask, pop))  # or: float(np.nansum(np.where(inside, pop.values, 0.0)))

        # Cropland km² within (area-true)
        if hasattr(cell_km2_est, "shape"):
            cl_km2_sum = float(np.nansum(np.where(inside, cl_frac.values * cell_km2_est, 0.0)))
            area_km2_within = float(np.nansum(np.where(inside, cell_km2_est, 0.0)))
        else:
            cl_km2_sum = float(np.nansum(np.where(inside, cl_frac.values, 0.0))) * float(cell_km2_est)
            area_km2_within = cells_within * float(cell_km2_est)

        # Electrified cells within (count of grid-present cells)
        elg_sum = float(np.count_nonzero(inside & (elg.values == 1)))

        # Shared denominators
        no_elec_flag = (elg_total == 0.0)

        # Append row (keep both current + legacy if you still want them)
        rows.append({
            "aoi": AOI,
            "travel_cut_min": float(thr),
            "pop_within": pop_sum,
            "cells_within": cells_within,
            "area_km2_within": area_km2_within,

            # % KPIs
            "pop_pct": (pop_sum / pop_total * 100.0) if pop_total > 0 else np.nan,
            "cropland_km2": cl_km2_sum,
            "cropland_pct": (cl_km2_sum / cl_km2_tot * 100.0) if cl_km2_tot > 0 else np.nan,
            "electrified_cells": elg_sum,
            "electrified_pct": (elg_sum / elg_total * 100.0) if elg_total > 0 else np.nan,

            # context
            "cell_area_km2": float(cell_km2_log),
            "note_electrification_denominator_zero": bool(no_elec_flag),

            # legacy/compat (remove if not needed)
            "threshold_min": float(thr),
            "pop_sum": pop_sum,
        })

        log.info(
            f"≤{thr:>3} min | cells={cells_within:,.0f} (area={area_km2_within:,.2f} km²) | "
            f"pop={pop_sum:,.0f} ({(pop_sum / pop_total * 100.0 if pop_total else np.nan):.1f}%) | "
            f"crop={cl_km2_sum:,.2f} km² "
            f"({(cl_km2_sum / cl_km2_tot * 100.0 if cl_km2_tot else np.nan):.1f}%) | "
            f"elec_cells={elg_sum:,.0f} "
            f"({(elg_sum / elg_total * 100.0 if elg_total else np.nan):.1f}%)"
        )


    # -------------------------------------------------------------------------
    # Sanity checks: KPI curves should be non-decreasing with larger thresholds
    # -------------------------------------------------------------------------
    kpis = pd.DataFrame(rows).sort_values("travel_cut_min")

    # Monotonic checks
    _assert_monotone_increasing(kpis["pop_within"].to_numpy(),       "Population within isochrone")
    _assert_monotone_increasing(kpis["cells_within"].to_numpy(),     "Cells within isochrone")
    _assert_monotone_increasing(kpis["area_km2_within"].to_numpy(),  "Area (km²) within isochrone")

    # AOI totals repeated per row (handy in spreadsheets)
    kpis["pop_total"] = pop_total
    kpis["cropland_total_km2"] = cl_km2_tot
    kpis["electrified_total_cells"] = elg_total

    # Optional: drop legacy duplicates if you don’t need them
    if "threshold_min" in kpis.columns and "travel_cut_min" in kpis.columns:
        # keep only the canonical column
        kpis = kpis.drop(columns=["threshold_min", "pop_sum"], errors="ignore")

    # Extra guardrails: keep % in [0,100] within tiny tolerance
    for c in ("pop_pct", "cropland_pct", "electrified_pct"):
        if c in kpis:
            kpis.loc[(kpis[c] < -1e-6) | (kpis[c] > 100 + 1e-6), c] = np.nan

    # -------------------------------------------------------------------------
    # Save table
    # -------------------------------------------------------------------------
    out_csv = KPI_CSV
    kpis.to_csv(out_csv, index=False)
    log.info(f"Saved KPI table → {out_csv}")

    log.info("Step 02 complete.")

if __name__ == "__main__":
    main()

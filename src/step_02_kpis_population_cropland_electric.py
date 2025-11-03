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
    PATHS, PARAMS, out_r, out_t, 
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

        # Guard against unexpected shape mismatch early
        _assert_same_shape(mask, pop)

        # Population within isochrone (sum where mask==1)
        pop_sum = zonal_sum(mask, pop)

        # Cropland km² within isochrone (area-true)
        if hasattr(cell_km2_est, "shape"):
            cl_km2_sum = float(
                np.nansum(np.where(mask.values == 1, cl_frac.values * cell_km2_est, np.nan))
            )
        else:
            cl_sum_frac = np.where(mask.values == 1, cl_frac.values, np.nan)
            cl_km2_sum = float(np.nansum(cl_sum_frac)) * float(cell_km2_est)

        # Electrified cells within isochrone (count)
        elg_sum = float(np.nansum((mask.values == 1) & (elg.values == 1)))

        # After computing elg_sum / elg_total and before rows.append(...)
        no_elec_flag = (elg_total == 0.0)

        rows.append({
            "threshold_min": thr,
            "pop_sum": pop_sum,
            "pop_pct": (pop_sum / pop_total * 100.0) if pop_total > 0 else np.nan,
            "cropland_km2": cl_km2_sum,
            "cropland_pct": (cl_km2_sum / cl_km2_tot * 100.0) if cl_km2_tot > 0 else np.nan,
            "electrified_cells": elg_sum,
            "electrified_pct": (elg_sum / elg_total * 100.0) if elg_total > 0 else np.nan,
            "cell_area_km2": cell_km2_log,
            "note_electrification_denominator_zero": bool(no_elec_flag),
        })

        log.info(
            f"≤{thr:>3} min | pop={pop_sum:,.0f} "
            f"({(pop_sum / pop_total * 100.0 if pop_total else np.nan):.1f}%) | "
            f"crop={cl_km2_sum:,.2f} km² "
            f"({(cl_km2_sum / cl_km2_tot * 100.0 if cl_km2_tot else np.nan):.1f}%) | "
            f"elec_cells={elg_sum:,.0f} "
            f"({(elg_sum / elg_total * 100.0 if elg_total else np.nan):.1f}%)"
        )

    # -------------------------------------------------------------------------
    # Sanity checks: KPI curves should be non-decreasing with larger thresholds
    # -------------------------------------------------------------------------
    kpis = pd.DataFrame(rows).sort_values("threshold_min")

    # Use sums for monotonicity (more numerically stable than %)
    _assert_monotone_increasing(kpis["pop_sum"].to_numpy(),          "Population within isochrone")
    _assert_monotone_increasing(kpis["cropland_km2"].to_numpy(),     "Cropland km² within isochrone")
    _assert_monotone_increasing(kpis["electrified_cells"].to_numpy(),"Electrified cells within isochrone")

    # Repeat denominators per row for spreadsheet clarity (optional)
    kpis["pop_total"]        = pop_total
    kpis["cropland_total_km2"] = cl_km2_tot
    kpis["electrified_total_cells"] = elg_total


    # -------------------------------------------------------------------------
    # Save table
    # -------------------------------------------------------------------------
    out_csv = KPI_CSV
    kpis.to_csv(out_csv, index=False)
    log.info(f"Saved KPI table → {out_csv}")

    log.info("Step 02 complete.")

if __name__ == "__main__":
    main()

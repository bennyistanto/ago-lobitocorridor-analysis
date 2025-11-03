"""
Step 01 — Isochrone masks (≤ thresholds in minutes).

Reads:
  - PATHS.TRAVEL (the travel-time raster; minutes)

Creates (AOI-prefixed), aligned 1-km GeoTIFFs:
  - {AOI}_iso_le_{thr}min_1km.tif   for each thr in PARAMS.ISO_THRESH

Notes:
  - Mask pixel values: 1 = inside the ≤thr isochrone; 0 = outside; NaN preserved.
  - Outputs are consumed by Step 02 (KPIs) and used in Step 06 (quick map).
"""

from __future__ import annotations
from typing import Iterable
import numpy as np

from config import PATHS, PARAMS, out_r, get_logger
from utils_geo import open_template, reclass_le, write_gtiff_masked

log = get_logger(__name__)


def main() -> None:
    """Build binary isochrone rasters at thresholds in PARAMS.ISO_THRESH."""
    # Load the travel-time raster (target grid)
    T = open_template(PATHS.TRAVEL)
    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"Isochrones from {PATHS.TRAVEL.name} | CRS={T.rio.crs} | "
        f"size={T.rio.height}x{T.rio.width} | cell={resx:.1f}x{resy:.1f}"
    )

    # Ensure thresholds are unique and sorted (minutes)
    thr_list = sorted(set(PARAMS.ISO_THRESH))
    if len(thr_list) != len(PARAMS.ISO_THRESH):
        log.info(f"De-duplicated ISO_THRESH; using {thr_list}")

    # Build & save each mask
    for thr in thr_list:
        log.info(f"Building ≤{thr} min isochrone...")
        # 1 where travel time ≤ thr, 0 otherwise, NaNs preserved
        mask = reclass_le(T, thr)
        out_path = out_r(f"iso_le_{thr}min_1km")
        write_gtiff_masked(mask, out_path, like=T, nodata=np.nan)

        # Quick coverage summary (useful sanity check)
        inside = int(np.nansum(mask.values == 1))
        total = int(np.isfinite(T.values).sum())
        pct = 100.0 * inside / total if total else np.nan
        log.info(f"Wrote {out_path.name} | cells inside: {inside}/{total} ({pct:.1f}%)")

    log.info("Step 01 complete.")


if __name__ == "__main__":
    main()

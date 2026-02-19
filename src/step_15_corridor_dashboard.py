"""
Step 15 — Corridor-wide Dashboard Table

Purpose
-------
Aggregate key metrics produced by earlier steps across ALL five AOIs into a
single, sortable summary table. This gives decision-makers a cross-province
comparison at a glance—answering all five research questions simultaneously.

How it works
------------
For each AOI directory found under outputs/tables, the step reads:
  - {AOI}_kpis_isochrones.csv         (Step 02 — pop/cropland/electrification by isochrone)
  - {AOI}_priority_muni_rank.csv      (Step 09 — municipality targeting score)
  - {AOI}_priority_clusters.csv       (Step 11 — cluster-level hotspots)
  - {AOI}_catchments_kpis.csv         (Step 12 — travel-time catchment beneficiaries)
  - {AOI}_od_gravity.csv              (Step 14 — OD flow summary)

It computes headline indicators per AOI and stacks them into one table.

Outputs
-------
- outputs/tables/corridor_dashboard.csv           (one row per AOI)
- outputs/tables/corridor_cluster_inventory.csv   (all clusters across AOIs)

Notes
-----
- Runs after all AOIs have completed Steps 0–14.
- Tolerates missing files: any AOI with partial outputs still gets a row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PATHS, get_logger, AOI

log = get_logger(__name__)

# Canonical AOI list (matches the 5 provinces along the Lobito Corridor)
CORRIDOR_AOIS = ("benguela", "bie", "huambo", "moxico", "moxicoleste")


def _safe_read(path: Path) -> pd.DataFrame | None:
    """Read CSV if it exists; return None otherwise."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.warning("Could not read %s: %s", path.name, e)
    return None


def _aoi_headline(aoi: str, tables_dir: Path) -> dict:
    """Extract headline metrics for one AOI from its output tables."""
    row: dict = {"aoi": aoi}

    # --- Step 02: Isochrone KPIs -----------------------------------------
    iso = _safe_read(tables_dir / f"{aoi}_kpis_isochrones.csv")
    if iso is not None and not iso.empty:
        # Typical columns: iso_thresh, pop_total, pop_in_iso, cropland_km2, ...
        # Take the broadest isochrone row for total-pop context
        if "pop_total" in iso.columns:
            row["pop_total"] = iso["pop_total"].iloc[0]
        if "iso_thresh" in iso.columns:
            for thr in (60, 120):
                sub = iso[iso["iso_thresh"] == thr]
                if not sub.empty:
                    if "pop_in_iso" in sub.columns:
                        row[f"pop_le{thr}min"] = float(sub["pop_in_iso"].iloc[0])
                    if "cropland_km2" in sub.columns:
                        row[f"cropland_km2_le{thr}min"] = float(sub["cropland_km2"].iloc[0])

    # --- Step 09: Municipality targeting ---------------------------------
    muni = _safe_read(tables_dir / f"{aoi}_priority_muni_rank.csv")
    if muni is not None and not muni.empty:
        row["n_municipalities"] = len(muni)
        row["muni_score_mean"] = float(muni["score"].mean()) if "score" in muni.columns else np.nan
        row["muni_score_max"] = float(muni["score"].max()) if "score" in muni.columns else np.nan
        if "pop_total" in muni.columns:
            row.setdefault("pop_total", float(muni["pop_total"].sum()))
        if "cropland_km2" in muni.columns:
            row["cropland_km2_total"] = float(muni["cropland_km2"].sum())
        if "area_km2" in muni.columns:
            row["area_km2_total"] = float(muni["area_km2"].sum())
        if "pct_electrified" in muni.columns:
            # Population-weighted mean electrification
            if "pop_total" in muni.columns:
                w = muni["pop_total"].fillna(0)
                if w.sum() > 0:
                    row["pct_electrified_popw"] = float(
                        (muni["pct_electrified"] * w).sum() / w.sum()
                    )
        if "poverty_rural" in muni.columns:
            row["poverty_rural_mean"] = float(muni["poverty_rural"].mean())

    # --- Step 11: Priority clusters --------------------------------------
    clust = _safe_read(tables_dir / f"{aoi}_priority_clusters.csv")
    if clust is not None and not clust.empty:
        row["n_clusters"] = len(clust)
        row["cluster_area_km2_total"] = float(clust["area_km2"].sum()) if "area_km2" in clust.columns else np.nan
        row["cluster_pop_total"] = float(clust["pop"].sum()) if "pop" in clust.columns else np.nan
        row["cluster_cropland_km2"] = float(clust["cropland_km2"].sum()) if "cropland_km2" in clust.columns else np.nan
        if "risk_roadcells" in clust.columns:
            row["cluster_risk_roadcells"] = int(clust["risk_roadcells"].sum())

    # --- Step 12: Catchment beneficiaries --------------------------------
    catch = _safe_read(tables_dir / f"{aoi}_catchments_kpis.csv")
    if catch is not None and not catch.empty:
        row["n_project_sites"] = int(catch["site_index"].nunique()) if "site_index" in catch.columns else np.nan
        # Aggregate at 60-min threshold (if present)
        c60 = catch[catch["thresh_min"] == 60] if "thresh_min" in catch.columns else pd.DataFrame()
        if not c60.empty and "pop" in c60.columns:
            row["catch60_pop_total"] = float(c60["pop"].sum())

    # --- Step 14: OD flows -----------------------------------------------
    od = _safe_read(tables_dir / f"{aoi}_od_gravity.csv")
    if od is not None and not od.empty:
        row["od_total_flow"] = float(od["flow"].sum()) if "flow" in od.columns else np.nan
        if "dist_km" in od.columns and "flow" in od.columns:
            w = od["flow"].values
            if w.sum() > 0:
                row["od_mean_dist_km"] = float(np.average(od["dist_km"].values, weights=w))

    return row


def _build_cluster_inventory(tables_dir: Path) -> pd.DataFrame:
    """Stack all per-AOI cluster tables into a single corridor-wide inventory."""
    frames = []
    for aoi in CORRIDOR_AOIS:
        fp = tables_dir / f"{aoi}_priority_clusters.csv"
        df = _safe_read(fp)
        if df is not None and not df.empty:
            df.insert(0, "aoi", aoi)
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    inv = pd.concat(frames, ignore_index=True)
    # Sort by population (desc) so the biggest hotspots surface first
    if "pop" in inv.columns:
        inv.sort_values("pop", ascending=False, inplace=True)
        inv.reset_index(drop=True, inplace=True)
    return inv


def main() -> None:
    tables_dir = PATHS.OUT_T

    # 1) Per-AOI headline dashboard
    rows = []
    for aoi in CORRIDOR_AOIS:
        log.info("Aggregating dashboard metrics for %s ...", aoi)
        rows.append(_aoi_headline(aoi, tables_dir))

    dashboard = pd.DataFrame(rows)

    # Append corridor-wide totals row
    totals = {"aoi": "CORRIDOR_TOTAL"}
    for col in dashboard.columns:
        if col == "aoi":
            continue
        vals = pd.to_numeric(dashboard[col], errors="coerce")
        if col.startswith("pct_") or col.endswith("_mean") or col.endswith("_max"):
            totals[col] = float(vals.mean())  # average for rates
        else:
            totals[col] = float(vals.sum())   # sum for counts / areas
    dashboard = pd.concat([dashboard, pd.DataFrame([totals])], ignore_index=True)

    out_dash = tables_dir / "corridor_dashboard.csv"
    dashboard.to_csv(out_dash, index=False)
    log.info("Wrote %s (%d rows)", out_dash.name, len(dashboard))

    # 2) Corridor-wide cluster inventory
    inv = _build_cluster_inventory(tables_dir)
    if not inv.empty:
        out_inv = tables_dir / "corridor_cluster_inventory.csv"
        inv.to_csv(out_inv, index=False)
        log.info("Wrote %s (%d clusters across all AOIs)", out_inv.name, len(inv))
    else:
        log.warning("No cluster tables found; skipping corridor cluster inventory.")

    log.info("Step 15 complete.")


if __name__ == "__main__":
    main()

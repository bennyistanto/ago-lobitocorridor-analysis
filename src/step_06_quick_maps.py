"""
Step 06 — Quick static PNG for briefings
Layers: Top 10% priority, isochrone rings (minutes), project sites (Map A)
        Road flood risk cells & near-priority risk (Map B)

Reads (AOI-prefixed / previous steps)
-------------------------------------
- PATHS.TRAVEL                 (target grid; minutes)  [grid only; not plotted]
- PRIORITY_TIF                 = {AOI}_priority_score_0_1.tif     [Step 03]
- PRIORITY_TOP10_TIF           = {AOI}_priority_top10_mask.tif    [Step 03]
- ROADS1K_TIF                  = {AOI}_roads_main_1km.tif         [Step 04]
- ROADS_RISK_TIF               = {AOI}_roads_flood_risk_cells_1km.tif     [Step 04]
- ROADS_RISK_NEAR_TIF          = {AOI}_roads_flood_risk_near_priority_1km.tif [Step 04]
- PATHS.SITES                  (project points; reprojected to grid CRS)
- PATHS.BND_ADM1               (AOI boundary polygon; plotted as thin outline)
- (optional) PATHS.ROADS       (OSM roads vector; light context in Map B)

Writes
------
- MAP_TRAVEL_PRIORITY_PNG      = outputs/figs/{AOI}_map_travel_priority.png    (Map A)
- outputs/figs/{AOI}_map_flood_bottlenecks.png                                  (Map B)

Notes
-----
- Travel time is *not* used as a background, to avoid masking key info.
- Map A communicates "where to focus": Priority heat (0–1), Top 10% outline, Isochrone rings (minutes), Sites.
- Map B communicates "what could break access": Road flood-risk cells (RP100 ≥ threshold) and which are near priority.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.enums import Resampling

from config import (
    AOI, PATHS, MAP_TRAVEL_PRIORITY_PNG,
    PRIORITY_TIF, PRIORITY_TOP10_TIF,
    ROADS1K_TIF, ROADS_RISK_TIF, ROADS_RISK_NEAR_TIF,
    out_f, out_r, get_logger
)
from utils_geo import open_template

log = get_logger(__name__)


def _assert_same_shape(*das):
    """Raise AssertionError if rasters do not share identical shape."""
    shapes = {da.shape for da in das}
    assert len(shapes) == 1, f"Rasters must share identical shape; got {shapes}"


def _ensure_match(da, T, name: str, resampling: Resampling):
    """
    Reproject `da` to match template `T` if shape/transform/CRS differ.
    Logs a warning once if a reproject occurs.
    """
    if (da.shape != T.shape or
        da.rio.transform() != T.rio.transform() or
        da.rio.crs != T.rio.crs):
        log.warning(f"Reprojecting {name} to match target grid")
        return da.rio.reproject_match(T, resampling=resampling)
    return da


def _xy_from_template(T):
    """Return imshow extent and meshgrid (X,Y) arrays for contour plotting in map coords."""
    x_min, x_max = float(T.x.min()), float(T.x.max())
    y_min, y_max = float(T.y.min()), float(T.y.max())
    X, Y = np.meshgrid(T.x.values, T.y.values)
    return (x_min, x_max, y_min, y_max), (X, Y)


def _plot_boundary(ax, bnd_gdf):
    """Thin boundary outline for geographic context without clutter."""
    try:
        if len(bnd_gdf):
            bnd_gdf.boundary.plot(ax=ax, linewidth=1.0, edgecolor="black", alpha=0.6, zorder=5)
    except Exception:
        pass


def _plot_sites(ax, gdf):
    """Plot project sites and (optionally) label them by `SiteID`."""
    if len(gdf):
        gdf.plot(ax=ax, markersize=18, edgecolor="k", facecolor="yellow", zorder=30)
        if "SiteID" in gdf.columns:
            for _, r in gdf.iterrows():
                ax.annotate(str(r["SiteID"]), (r.geometry.x, r.geometry.y),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color="k", zorder=31)


def _try_read_isochrones(T):
    """Try to load ≤30/60/120/240 minute isochrone masks if present; return list of (thr, mask)."""
    masks = []
    for thr in (30, 60, 120, 240):
        p = out_r(f"iso_le_{thr}min_1km")
        try:
            m = open_template(p)
            m = _ensure_match(m, T, f"iso_{thr}", Resampling.nearest)
            masks.append((thr, m))
        except Exception:
            continue
    return masks


def _plot_isochrones(ax, X, Y, iso_masks):
    """Contour the binary masks as thin rings (≤thr) if available; labeled in legend."""
    for thr, m in iso_masks:
        try:
            cs = ax.contour(X, Y, m.values, levels=[0.5], linewidths=1.0, alpha=0.8, zorder=15)
            for c in cs.collections:
                c.set_label(f"≤{thr} min")
        except Exception:
            continue


def map_a_access_priority():
    """
    Map A — Access & Priority (no background):
    - Priority heat overlay (0..1, semi-transparent)
    - Top 10% priority outline (bold)
    - Isochrone rings (≤30/60/120/240 minutes)
    - AOI boundary outline
    - Project sites with labels
    """
    # Load template & rasters; ensure grid match
    T      = open_template(PATHS.TRAVEL)   # used only for grid/extent
    prio   = open_template(PRIORITY_TIF)
    mask10 = open_template(PRIORITY_TOP10_TIF)
    prio   = _ensure_match(prio, T, "priority", Resampling.bilinear)
    mask10 = _ensure_match(mask10, T, "priority_top10", Resampling.nearest)
    _assert_same_shape(T, prio, mask10)

    tf = T.rio.transform()
    resx, resy = abs(tf.a), abs(tf.e)
    log.info(
        f"Map A inputs | CRS={T.rio.crs} | size={T.rio.height}x{T.rio.width} | cell={resx:.4f}x{resy:.4f}"
    )

    # Layers
    sites = gpd.read_file(PATHS.SITES).to_crs(T.rio.crs)
    bnd   = gpd.read_file(PATHS.BND_ADM1).to_crs(T.rio.crs)
    iso_masks = _try_read_isochrones(T)
    log.info(f"Map A: sites={len(sites)} | isochrone_masks_loaded={len(iso_masks)}")

    # Coordinates
    (x_min, x_max, y_min, y_max), (X, Y) = _xy_from_template(T)

    # Plot (white background; no basemap)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_facecolor("white")

    # Priority heat (semi-transparent; communicates 0..1 range)
    im_pr = ax.imshow(prio.values, extent=[x_min, x_max, y_min, y_max],
                      origin="upper", alpha=0.55, zorder=10)
    cbar = plt.colorbar(im_pr, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Priority score (0–1)")

    # Top 10% outline (bold)
    ax.contour(X, Y, mask10.values, levels=[0.5], linewidths=1.8, alpha=0.95, colors="k", zorder=20)

    # Isochrone rings (minutes)
    if iso_masks:
        _plot_isochrones(ax, X, Y, iso_masks)

    # Boundary & sites
    _plot_boundary(ax, bnd)
    _plot_sites(ax, sites)

    # Legend (explicit units)
    handles = [
        Line2D([0], [0], color="k", lw=2, label="Top 10% priority"),
        Patch(facecolor="none", edgecolor="k", label="Isochrone (minutes: ≤30/60/120/240)"),
        Line2D([0], [0], marker="o", color="k", markerfacecolor="yellow", markersize=8, lw=0, label="Project site")
    ]
    ax.legend(handles=handles, loc="lower right")

    # Title + explanatory subtitle (what “Top 10%” means; isochrone units)
    ax.set_title(f"{AOI.title()} — Priority Focus Areas", pad=10)
    ax.text(0.01, 0.01,
            "Top 10% = highest composite priority (0–1) across access, population, vegetation, night lights, and drought.\n"
            "Isochrone rings labeled in minutes (≤30/60/120/240).",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig(MAP_TRAVEL_PRIORITY_PNG, dpi=300)
    log.info(f"Saved Map A → {MAP_TRAVEL_PRIORITY_PNG}")


def map_b_flood_bottlenecks():
    """
    Map B — Flood Bottlenecks (RP100 threshold; no background):
    - Road flood-risk cells (roads ∧ flood ≥ threshold)
    - Flood-risk near priority (accented/darker)
    - Top 10% priority outline (context)
    - AOI boundary outline
    - Main roads (light gray), Project sites
    """
    # Load template & rasters; ensure grid match
    T       = open_template(PATHS.TRAVEL)   # grid only; not plotted
    risk    = open_template(ROADS_RISK_TIF)
    risk_np = open_template(ROADS_RISK_NEAR_TIF)
    prio10  = open_template(PRIORITY_TOP10_TIF)
    roads1k = open_template(ROADS1K_TIF)

    risk    = _ensure_match(risk,    T, "roads_risk", Resampling.nearest)
    risk_np = _ensure_match(risk_np, T, "roads_risk_near_priority", Resampling.nearest)
    prio10  = _ensure_match(prio10,  T, "priority_top10", Resampling.nearest)
    roads1k = _ensure_match(roads1k, T, "roads_main", Resampling.nearest)
    _assert_same_shape(T, risk, risk_np, prio10, roads1k)

    sites = gpd.read_file(PATHS.SITES).to_crs(T.rio.crs)
    bnd   = gpd.read_file(PATHS.BND_ADM1).to_crs(T.rio.crs)
    log.info(f"Map B: sites={len(sites)}")

    # Coordinates
    (x_min, x_max, y_min, y_max), (X, Y) = _xy_from_template(T)

    # Plot (white background; focus on risk)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.set_facecolor("white")

    # Main roads presence (light gray, faint context)
    ax.imshow(np.where(roads1k.values == 1, 1, np.nan),
              extent=[x_min, x_max, y_min, y_max], origin="upper",
              alpha=0.25, cmap="Greys", zorder=6)

    # Risk cells (red; visible)
    ax.imshow(np.where(risk.values == 1, 1, np.nan),
              extent=[x_min, x_max, y_min, y_max], origin="upper",
              alpha=0.65, cmap="Reds", zorder=12)

    # Risk near priority (darker red; most important)
    ax.imshow(np.where(risk_np.values == 1, 1, np.nan),
              extent=[x_min, x_max, y_min, y_max], origin="upper",
              alpha=0.9, cmap="Reds", zorder=13)

    # Top 10% priority outline (context)
    ax.contour(X, Y, prio10.values, levels=[0.5], linewidths=1.4, colors="k", zorder=20)

    # Boundary & sites
    _plot_boundary(ax, bnd)
    _plot_sites(ax, sites)

    # Legend (explicit meaning)
    handles = [
        Line2D([0], [0], color="k", lw=1.4, label="Top 10% priority (outline)"),
        Patch(facecolor="gray", alpha=0.25, label="Main roads (context)"),
        Patch(facecolor="red",  alpha=0.65, label="Road flood-risk cell (RP100 ≥ threshold)"),
        Patch(facecolor="red",  alpha=0.90, label="Flood-risk near priority"),
        Line2D([0], [0], marker="o", color="k", markerfacecolor="yellow", markersize=8, lw=0, label="Project site")
    ]
    ax.legend(handles=handles, loc="lower right")

    # Title + explanatory subtitle (what “near priority” means)
    ax.set_title(f"{AOI.title()} — Flood Bottlenecks (No Basemap)", pad=10)
    ax.text(0.01, 0.01,
            "Red cells = main road intersects RP100 flood depth ≥ threshold.\n"
            "Darker red = those risk cells within ~1 km of Top 10% priority areas.",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    plt.tight_layout()

    out_png = out_f("map_flood_bottlenecks")
    plt.savefig(out_png, dpi=300)
    log.info(f"Saved Map B → {out_png}")


def main() -> None:
    """Generate the two communication maps (Priority focus; Flood bottlenecks)."""
    map_a_access_priority()
    map_b_flood_bottlenecks()
    log.info("Step 06 complete.")


if __name__ == "__main__":
    main()

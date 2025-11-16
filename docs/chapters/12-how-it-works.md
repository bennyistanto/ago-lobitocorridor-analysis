# 12. How it works?

## Problem

Stakeholders need a clear, non-code explanation of **what the pipeline does**, how layers combine into a **priority surface**, and how later chapters derive clusters, rankings, catchments, synergies, and OD flows.

## Strategy

Describe the pipeline as a **four-layer sandwich**:

1. **Inputs & alignment** (Step 00) → clean, AOI-consistent 1-km stack.
2. **Priority & selection** (Steps 07/10) → composite score + Top-X envelope.
3. **Actionable products** (Steps 11–13) → clusters, catchments, synergies.
4. **Mobility context** (Step 14) → OD-Lite flows for corridor movement.

Keep equations intuitive, knobs transparent, and outputs reproducible.

## Data

* Core rasters: **travel time to market (minutes)**, **WorldPop (people)**, **cropland fraction** (10 m → 1 km), **night-time lights**, **veg index**, **drought frequency**, **flood depth (30 m → 1 km)**.
* Core vectors: **roads/rail**, **electrification & settlement masks**, **project sites**, **Admin2 (RAPP themes)**.
* All aligned to an AOI 1-km grid in **EPSG:4326**.

## Methods (brief, step by step)

* **Step 00 – Align & rasterize.**
  Reproject/align each raster to the **same 1-km grid**; rasterize vectors (cropland, electrification, rural/urban). Aggregate flood **30 m → 1 km (max)**. Write `{AOI}_*_1km.tif`.
* **Step 07 – Priority (tunable).**
  Build a **priority surface** on [0,1]:
  *Normalize + weight* ⇒ `score = W_POP·f(pop) + W_VEG·f(veg) + W_NTL·f(ntl) + W_DRT·g(drought) + …`
  Apply **masks/thresholds** (e.g., rural-only mask and minimum cropland fraction) before/after scoring.
* **Step 10 – Selection & scenarios.**
  Choose **Top-%** *or* **Top-km²** cells; optionally run **scenarios** (e.g., drop NTL/VEG, higher cropland threshold) and summarize **stability/swing** across municipalities.
* **Step 11 – Clusters & KPIs.**
  Label connected components (8-neighbour) inside the Top-X mask; prune small blobs; compute **area, population, cropland km², % electrified, mean travel time, mean drought, dominant Admin2** per cluster.
* **Step 12 – Catchments.**
  From each **project site**, compute **≤30/60/120 min** isochrones over the minutes raster; aggregate **people/cropland** per band, plus marginal gains.
* **Step 13 – Synergies.**
  For **sites & clusters**, compute **nearest distance** and **counts within 5/10/30 km** to Gov/WB/Other projects → coordination opportunities.
* **Step 14 – Origin-Destination.**
  A simple **gravity model** on Admin2 zones (population × opportunities × distance-decay) to highlight **desire lines** and **hub municipalities** (inflow/outflow).

> All chapters simply **load** what these steps write into `/outputs`.

## Outputs

* Rasters: `outputs/rasters/{AOI}_*.tif` (aligned 1-km stack, selection mask, cluster labels).
* Tables: `outputs/tables/{AOI}_*.csv` (municipality ranks, clusters KPIs, catchments, synergies, scenarios, OD flows).

---

## How to run (analyst)

**This cell prints the active parameter subset that shapes the priority surface (read-only).**

```{code-cell} ipython3
import os, sys, pprint
from pathlib import Path
from dataclasses import asdict

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
sys.path.append(str(ROOT / "src"))

from config import PARAMS

pp = pprint.PrettyPrinter(width=100, compact=True)
params = asdict(PARAMS)

keys = [
    "W_ACC","W_POP","W_VEG","W_NTL","W_DRT",
    "MASK_REQUIRE_RURAL","MASK_MIN_CROPLAND",
    "SMOOTH_RADIUS",
    "TOP_PCT_CELLS","TOP_KM2",
    "MIN_CLUSTER_CELLS",
    "SYNERGY_RADII_KM",
    "W_POV","W_FOOD","W_MTT","W_RWI",
]

print("AOI:", AOI)
pp.pprint({k: params.get(k) for k in keys})
```

**This cell lists each step and the headline files it produces (sanity checklist).**

```{code-cell} ipython3
steps = [
    ("00 Align & rasterize", [
        "{AOI}_pop_1km.tif",
        "{AOI}_cropland_fraction_1km.tif",
        "{AOI}_flood_rp100_maxdepth_1km.tif",
    ]),
    ("07 Priority (tunable)", [
        "{AOI}_priority_score_1km.tif",
        "{AOI}_priority_admin2_rank.csv",
        "{AOI}_priority_muni_rank.csv",
    ]),
    ("10 Scenario summary", [
        "{AOI}_priority_scenarios_summary.csv",
    ]),
    ("11 Clusters & KPIs", [
        "{AOI}_priority_top10_mask.tif", 
        "{AOI}_priority_clusters_1km.tif",
        "{AOI}_priority_clusters.csv",
    ]),
    ("12 Catchments", [
        "{AOI}_catchments_kpis.csv",
    ]),
    ("13 Synergies", [
        "{AOI}_site_synergies.csv",
        "{AOI}_cluster_synergies.csv",
    ]),
    ("14 OD-Lite", [
        "{AOI}_od_gravity.csv",
        "{AOI}_od_zone_attrs.csv",
        "{AOI}_od_agents.csv",
    ]),
]

for name, files in steps:
    print(f"{name}:")
    for f in files:
        print("  - outputs/.../", f.replace("{AOI}", AOI))
```

---

## How the priority score is built (intuition)

1. **Normalize** each layer to [0,1]. For benefits (pop, veg, ntl) higher is better. For risks (drought), invert or apply a dampening transform `g(·)`.
2. **Weight** layers using `W_*` to reflect importance.
3. **Mask/threshold** to focus on rural, unelectrified, and minimal cropland contexts.
4. **Select** either the top **percentage** of cells or a **fixed km²** envelope.

**This cell echoes the core formula using your current weights (for communication, not math enforcement).**

```{code-cell} ipython3
from dataclasses import asdict

p = asdict(PARAMS)
w = {k: v for k, v in p.items() if k.startswith("W_")}

expr = "score ≈ " + " + ".join(
    f"{w[k]}·{k[2:].lower()}" for k in sorted(w)
)
print(expr)

print("Masks/thresholds:", {
    "require_rural": p.get("MASK_REQUIRE_RURAL"),
    "min_cropland": p.get("MASK_MIN_CROPLAND"),
})

print("Selection:", {
    "top_pct_cells": p.get("TOP_PCT_CELLS"),
    "top_km2": p.get("TOP_KM2"),
})

print("Smoothing radius (cells):", p.get("SMOOTH_RADIUS"))
```

---

## How to read the results (interpretation)

* Think of the **priority surface** as a **heatmap of opportunity**: combining **need (pop, rural, unelectrified)** and **potential (cropland, access)**, optionally tempered by **risk**.
* **Selection envelope** (Top-% or Top-km²) defines the **actionable area**; **clusters** convert pixels into **projects**.
* **Catchments** translate project siting into **people reached per minute**; **synergies** show **delivery leverage**; **OD-Lite** connects it to **movement patterns**.

## Caveats

* Public/EO layers are **proxies**; they are powerful for triage, not a replacement for field engineering.
* Changing **zone design** (Admin2 vs. grids) or **selection envelope** can shift rankings—use scenarios to show robustness.
* CRS and resolution must **match** across rasters; stale outputs can cause misalignment (use Chapter 11 checks).

### Download

Nothing to download—this is a **method note**. Use earlier chapters for artifact downloads.

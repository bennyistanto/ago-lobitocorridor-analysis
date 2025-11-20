# 1. If we could only start in three places…

## Problem

Where should we focus the **first wave** of last-mile investments along the Lobito Corridor, given limited funds?

## Strategy

Use a transparent **priority surface** (access + people + production + simple risk screens), then group the top cells into **actionable clusters** and rank them.

## Data

- **Travel time to market** (GOST modelled access 1 km)
- **Population** (WorldPop, 1 km)  
- **Cropland fraction** (from WorldCover, 10 m → 1 km)  
- **Rural/urban mask, electrification** (WBG Global Electrification Platform)
- *(Optional)* **Drought** (FAO's ASI 1 km), 
- *(Optional)* **Flood depth** (FATHOM Pluvial 100yr 30 m → 1 km)

## Methods (brief)

- **Step 07 / 10**: build priority score (0–1), apply masks & thresholds, select **Top-X** (% or km²).  
- **Step 11**: label **clusters inside the Top-X mask** (8-neigh), prune small areas; compute KPIs (area, pop, cropland km², % electrified, mean travel time, drought); tag dominant municipality.  
- **Step 12** (optional context): site catchments (≤30/60/120 min) from a friction surface.

## Outputs

- `outputs/tables/{AOI}_priority_clusters.csv` — one row per cluster (KPIs)  
- `outputs/rasters/{AOI}_priority_top10_mask.tif` — binary selection (exact name per Step 10 config)  
- `outputs/rasters/{AOI}_priority_clusters_1km.tif` — labeled clusters

## How to run (analyst)

Run: **07 → 10 (optional scenarios) → 11 (clusters)** once. This chapter only **loads** saved outputs (no recomputation).

**This cell loads the cluster KPIs table from `/outputs` for the current AOI (no recomputation).**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "huambo")
OUT_T = ROOT / "outputs" / "tables"

clusters = pd.read_csv(OUT_T / f"{AOI}_priority_clusters.csv")
clusters.head(10)
```

## How to read the results

- Look for **large clusters** with **high pop & cropland**, **low electrification**, and **reasonable access** (mean minutes not extreme). These are strong candidates for a first tranche.
- If clusters look too sparse or tiny, increase MIN_CLUSTER_CELLS or switch from Top-% to Top-km² in Step 10.

## Caveats

- Priority is a proxy built from public/EO layers; always confirm with field intel.
- If NTL/VEG bias the score, use scenarios in Chapter 4 to test robustness.

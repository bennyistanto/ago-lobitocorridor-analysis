# 3. Where are the actionable hotspots?

## Problem

Which specific **places** (not entire municipalities) are most ready for a first wave of last-mile investments—large enough to matter, reachable, and rich in smallholder potential?

## Strategy

Turn the corridor-wide **priority surface** into **connected clusters** of top-scoring cells. Keep clusters that are big enough to act on, then quantify what’s inside (people, cropland, electrification, basic access, risk screens).

## Data

* **Priority surface (0–1, 1-km)** — Step 07/10 result
* **Top-X selection mask (1-km)** — Step 10 (Top-% *or* Top-km²)
* **Cluster labels (1-km)** — Step 11
* **Population (WorldPop, 1-km)** — Step 00
* **Cropland fraction (10 m → 1-km)** — Step 00
* **Electrification / Rural mask (WBG)** — Step 00
* *(Optional screens)* Drought (1-km), flood depth (30 m → 1-km)

## Methods (brief)

* **Select:** from the priority raster, keep **Top-X** cells by *percentage* or *fixed km²*.
* **Clean:** remove speckles by **cell count** and **km²** minimums; (optional) Gaussian smooth before selection.
* **Label:** 8-neighbour connected components → **clusters**.
* **Summarize:** for each cluster compute **area (km²)**, **population**, **cropland km²**, **% electrified**, **mean travel time**, **mean drought**, and **dominant municipality**.

(Produced by Step 11. See Appendix for parameter names.)

## Outputs

* `outputs/tables/{AOI}_priority_clusters.csv` — one row per cluster (KPIs & dominant municipality)
* `outputs/rasters/{AOI}_priority_clusters_1km.tif` — labeled clusters (0 = background)
* `outputs/rasters/{AOI}_priority_top10_mask.tif` — binary selection used to define clusters

## How to run (analyst)

Run Steps **07 → 10 → 11** once. This chapter only loads saved outputs (no recomputation).

**This cell loads the cluster KPIs table from `/outputs` for the current AOI.**

```python
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT_T = ROOT / "outputs" / "tables"

clusters = pd.read_csv(OUT_T / f"{AOI}_priority_clusters.csv")
clusters.head(10)
```

## Quick results

**This cell shows the top-10 clusters by population (headline view).**

```python
top_pop = (clusters
           .sort_values("pop", ascending=False)
           .head(10)[["cluster_id","NAM_2","area_km2","pop","cropland_km2","pct_electrified"]])
print("Top-10 clusters by population")
top_pop
```

**This cell shows the top-10 clusters by cropland area (km²).**

```python
top_crop = (clusters
            .sort_values("cropland_km2", ascending=False)
            .head(10)[["cluster_id","NAM_2","area_km2","pop","cropland_km2","pct_electrified"]])
print("Top-10 clusters by cropland km²")
top_crop
```

**This cell summarizes which municipalities host the most/largest clusters.**

```python
cov = (clusters
       .groupby("NAM_2", as_index=False)
       .agg(clusters=("cluster_id","count"),
            area_km2=("area_km2","sum"),
            pop=("pop","sum"),
            cropland_km2=("cropland_km2","sum"))
       .sort_values(["clusters","pop"], ascending=[False, False]))
cov.head(10)
```

*(Optional tiny chart—safe to skip if you want a text-only book.)*
**This cell draws a quick bar chart of the top 8 clusters by population.**

```python
import matplotlib.pyplot as plt

t8 = clusters.nlargest(8, "pop")
plt.figure()
plt.barh([f"#{cid} • {n2}" for cid,n2 in zip(t8["cluster_id"], t8["NAM_2"])], t8["pop"])
plt.gca().invert_yaxis()
plt.xlabel("People (WorldPop)")
plt.title(f"{AOI}: Top clusters by population")
plt.show()
```

## How to read the results (interpretation)

* **Big & dense beats big & empty.** Prefer clusters with high **pop** and **cropland km²**, not just large area.
* **Low electrification = opportunity.** A lower **% electrified** inside a high-pop cluster flags places where power/social services can multiply benefits.
* **Access matters.** Extremely high **mean travel time** (inside the cluster) suggests harder delivery—pair with a road fix or start elsewhere.
* **Municipality alignment.** Use the coverage table to see which **NAM\_2** host most of the promising clusters (cross-check with Chapter 2).

## Caveats

* Clusters are **1-km pixels** glued together; edges are schematic, not parcel boundaries.
* WorldPop and cropland fractions are **modelled**; validate with local data before committing sites.
* If clusters look too fragmented: increase `MIN_CLUSTER_CELLS` or switch selection to **Top-km²** in Step 10.

### Tweak without code edits (recap)

* **Selection envelope:** switch `TOP_PCT_CELLS` ↔ `TOP_KM2`.
* **Coherence:** raise `MIN_CLUSTER_CELLS` / `MIN_CLUSTER_KM2`.
* **Focus:** raise `MASK_MIN_CROPLAND` to target stronger farming zones.
* **Signal mix:** adjust **W\_POP** (up) and **W\_DRT/NTL/VEG** (down) if proxies dominate.

### Download

* Clusters **table** → `outputs/tables/{AOI}_priority_clusters.csv`
* Cluster **raster** → `outputs/rasters/{AOI}_priority_clusters_1km.tif`
* Selection **mask** → `outputs/rasters/{AOI}_priority_top10_mask.tif`

---

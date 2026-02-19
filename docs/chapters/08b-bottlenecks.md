---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# 8b. Which road segments carry the most risk?

## The decision question

If a flood damages one road segment, which disruption would hurt the most people? Knowing this helps decision makers **prioritize road resilience investments** where they protect the largest share of corridor movement.

## What this analysis does

Step 14 already models OD flows between municipalities using a gravity model. This extension takes those flows and asks: *which physical road cells do they cross?* By overlaying the flow desire-lines on the flood-risk road cells from Step 04, we can rank every at-risk road cell by the **total OD flow it carries**.

A high-flow, high-risk cell is a **critical bottleneck**: if it fails during a flood, a large share of corridor interaction is disrupted.

## How it works (non-technical)

1. For each municipality-to-municipality flow, draw a straight line between centroids
2. Mark every 1-km grid cell that line crosses
3. Add up the flow from all OD pairs passing through each cell
4. Intersect with flood-risk road cells (from Step 04)
5. Rank the risk cells from highest to lowest accumulated flow

The result is a prioritized list of road segments where flood-proofing investments would protect the most movement.

## Data

* **OD gravity flows** — `outputs/tables/{AOI}_od_gravity.csv` (Step 14)
* **Flood-risk road cells** — `outputs/rasters/{AOI}_roads_flood_risk_cells_1km.tif` (Step 04)
* **Zone centroids** — `outputs/tables/{AOI}_od_zone_attrs.csv` (Step 14)

## Outputs

* `outputs/tables/{AOI}_od_bottleneck_cells.csv` — each flood-risk road cell ranked by OD flow load
* `outputs/rasters/{AOI}_od_bottleneck_risk.tif` — flow load raster on risk cells only

## How to run (analyst)

Run **Step 14** (which now includes the bottleneck overlay automatically). This chapter loads saved outputs.

**This cell loads the bottleneck ranking table.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")
OUT_T = ROOT / "outputs" / "tables"

btl_path = OUT_T / f"{AOI}_od_bottleneck_cells.csv"
btl = pd.read_csv(btl_path) if btl_path.exists() else None

if btl is not None:
    print(f"Loaded {len(btl)} risk cells with OD flow")
    btl.head(15)
else:
    print("Bottleneck table not found; run Step 14.")
```

## Quick results

**This cell shows the top-20 most critical bottleneck cells.**

```{code-cell} ipython3
if btl is not None:
    top20 = btl.head(20)[["risk_rank", "lon", "lat", "flow_load"]]
    print("Top-20 flood-risk road cells by OD flow load")
    top20
```

**This cell draws a bar chart of the top 10 bottleneck cells.**

```{code-cell} ipython3
import matplotlib.pyplot as plt

if btl is not None:
    t10 = btl.head(10)
    labels = [f"({row.lon:.3f}, {row.lat:.3f})" for _, row in t10.iterrows()]
    plt.figure()
    plt.barh(labels, t10["flow_load"])
    plt.gca().invert_yaxis()
    plt.xlabel("Total OD flow through cell")
    plt.title(f"{AOI}: Top bottleneck road cells (flood-risk)")
    plt.show()
```

## How to read the results

* **Top-ranked cells** represent the road segments where a flood event would disrupt the most inter-municipal movement. These are strong candidates for bridge upgrades, drainage improvements, or alternative route planning.
* **Flow load** is relative (based on the gravity model's total trip scaling), so compare cells within the same AOI rather than across provinces.
* **Clustered bottleneck cells** (nearby cells with high flow) often indicate a single critical road link rather than scattered risks.

## Caveats

* Desire lines are **straight-line approximations** between centroids, not actual road routes. Cells near the centroid-to-centroid line may not be the actual road used.
* Gravity flows are **modeled**, not observed. The ranking reflects interaction potential, not measured traffic.
* For planning-grade bottleneck analysis, combine this screen with road engineering assessments and actual traffic counts.

### Download

* Bottleneck **table** → `outputs/tables/{AOI}_od_bottleneck_cells.csv`
* Bottleneck **raster** → `outputs/rasters/{AOI}_od_bottleneck_risk.tif`

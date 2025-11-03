# 6. Stacking with other investments (synergies)

## Problem

Where can we **piggy-back** on ongoing or planned investments (Government, World Bank, other partners) so last-mile upgrades land where coordination is strongest and delivery is faster?

## Strategy

Measure simple, decision-friendly synergies: **nearest distance** to other projects and **counts within 5/10/30 km** for both **project sites** and **priority clusters**. Use these to flag easy coordination wins and avoid isolated bets.

## Data

* **Project registries** (Gov / WB / Other) — point layers prepared for AOI
* **Project sites** — Diversifica Mais / AOI sites (points)
* **Priority clusters** — Step 11 (polygons derived from 1-km labels)
* *(Optional)* Admin2 boundaries for reporting

## Methods (brief)

* **Step 13** computes:

  * For each **site**: nearest **Gov/WB/Other** distance (km) and project **counts within 5/10/30 km**.
  * For each **cluster** (by centroid or polygon edge): same set of metrics.
* Outputs are tidy CSVs keyed by `site_id` or `cluster_id`.

## Outputs

* `outputs/tables/{AOI}_synergy_sites.csv`
  *(e.g., columns: site\_id, dist\_gov\_km, dist\_wb\_km, dist\_oth\_km, n5\_gov, n5\_wb, n5\_oth, n10\_*, n30\_\* …)\*
* `outputs/tables/{AOI}_synergy_clusters.csv`
  *(e.g., columns: cluster\_id, NAM\_2, dist\_gov\_km, …, n30\_oth …)*

## How to run (analyst)

Run **Step 13** once. This chapter only **loads** saved outputs (no recomputation).

**This cell loads the site-level and cluster-level synergy tables for the current AOI.**

```python
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT_T = ROOT / "outputs" / "tables"

syn_sites    = pd.read_csv(OUT_T / f"{AOI}_synergy_sites.csv")
syn_clusters = pd.read_csv(OUT_T / f"{AOI}_synergy_clusters.csv")
syn_sites.head(10), syn_clusters.head(10)
```

## Quick results

**This cell lists Top-10 sites with the **closest** World Bank project (coordination quick wins).**

```python
cols_needed = {"site_id","dist_wb_km"}
if cols_needed.issubset(syn_sites.columns):
    top_close_wb = syn_sites.sort_values("dist_wb_km").head(10)[["site_id","dist_wb_km"]]
    top_close_wb
else:
    print("Expected columns missing in syn_sites (need site_id, dist_wb_km).")
```

**This cell lists Top-10 clusters with the **largest number of projects within 10 km** (all partners).**

```python
cand_cols = [c for c in syn_clusters.columns if c.startswith("n10_")]
if "cluster_id" in syn_clusters.columns and cand_cols:
    syn_clusters["n10_total"] = syn_clusters[cand_cols].sum(axis=1)
    top_cov = (syn_clusters
               .sort_values("n10_total", ascending=False)
               .head(10)[["cluster_id","NAM_2","n10_total"] + cand_cols])
    top_cov
else:
    print("Expected columns missing in syn_clusters (need cluster_id, n10_*).")
```

**This cell flags sites with **zero projects within 30 km** (islands to treat with caution or to seed first).**

```python
c30 = [c for c in syn_sites.columns if c.startswith("n30_")]
if c30:
    islands = syn_sites.loc[syn_sites[c30].sum(axis=1) == 0, ["site_id"] + c30]
    islands.head(20)
else:
    print("No n30_* columns found in syn_sites.")
```

**This cell ranks municipalities by **density of nearby projects** around clusters (who can coordinate most).**

```python
if {"NAM_2","cluster_id"}.issubset(syn_clusters.columns) and cand_cols:
    muni_proj = (syn_clusters
                 .assign(n10_total=lambda d: d[cand_cols].sum(axis=1))
                 .groupby("NAM_2", as_index=False)
                 .agg(clusters=("cluster_id","count"),
                      n10_total=("n10_total","sum"))
                 .sort_values(["n10_total","clusters"], ascending=[False, False]))
    muni_proj.head(10)
else:
    print("NAM_2 or cluster_id missing in syn_clusters.")
```

*(Optional tiny chart—safe to skip if you want a text-only book.)*
**This cell draws a quick bar of Top-8 sites by “projects within 10 km (all partners)”.**

```python
import matplotlib.pyplot as plt

c10 = [c for c in syn_sites.columns if c.startswith("n10_")]
if c10:
    syn_sites["n10_total"] = syn_sites[c10].sum(axis=1)
    t8 = syn_sites.nlargest(8, "n10_total")
    plt.figure()
    plt.barh(t8["site_id"].astype(str), t8["n10_total"])
    plt.gca().invert_yaxis()
    plt.xlabel("Projects within 10 km (Gov + WB + Other)")
    plt.title(f"{AOI}: Sites with strongest coordination potential")
    plt.show()
```

## How to read the results (interpretation)

* **Closest WB/Gov projects** → easiest **coordination wins** (shared supervision, shared contractors, faster mobilization).
* **High project counts within 10–30 km** → areas with **institutional presence**; stack last-mile upgrades here for speed and synergy.
* **Islands** (zero projects within 30 km) → may still be strategic, but expect **higher delivery friction**; plan more groundwork.
* **Municipality view** helps align with local authorities and avoid duplications.

## Caveats

* Registries can be **incomplete or lagged**—confirm with PIUs and provincial directorates.
* Distances are **straight-line** unless Step 13 used road distances; treat as screening, not final logistics.
* If clusters overlap municipal borders, aggregations by **NAM\_2** reflect centroid assignment (check edge cases if critical).

### Download

* Synergy **sites** → `outputs/tables/{AOI}_synergy_sites.csv`
* Synergy **clusters** → `outputs/tables/{AOI}_synergy_clusters.csv`

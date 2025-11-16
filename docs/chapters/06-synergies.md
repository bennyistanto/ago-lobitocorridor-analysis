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

* `outputs/tables/{AOI}_site_synergies.csv`  
  *(columns: site_id, lon, lat, dist_km_nearest_gov, dist_km_nearest_wb, dist_km_nearest_oth, and for each radius r in SYNERGY_RADII_KM: count_gov_le{r}km, count_wb_le{r}km, count_oth_le{r}km)*

* `outputs/tables/{AOI}_cluster_synergies.csv`  
  *(columns: cluster_id, lon, lat, dist_km_nearest_gov, dist_km_nearest_wb, dist_km_nearest_oth, plus the same count_gov_le{r}km / count_wb_le{r}km / count_oth_le{r}km fields)*


## How to run (analyst)

Run **Step 13** once. This chapter only **loads** saved outputs (no recomputation).

**This cell loads the site-level and cluster-level synergy tables for the current AOI.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT_T = ROOT / "outputs" / "tables"

syn_sites    = pd.read_csv(OUT_T / f"{AOI}_site_synergies.csv")
syn_clusters = pd.read_csv(OUT_T / f"{AOI}_cluster_synergies.csv")
syn_sites.head(10), syn_clusters.head(10)
```

## Quick results

**This cell lists Top-10 sites with the **closest** World Bank project (coordination quick wins).**

```{code-cell} ipython3
cols_needed = {"site_id","dist_km_nearest_wb"}
if cols_needed.issubset(syn_sites.columns):
    top_close_wb = syn_sites.sort_values("dist_km_nearest_wb").head(10)[["site_id","dist_km_nearest_wb"]]
    top_close_wb
else:
    print("Expected columns missing in syn_sites (need site_id, dist_km_nearest_wb).")
```

**This cell lists Top-10 clusters with the **largest number of projects within 10 km** (all partners).**

```{code-cell} ipython3
cand_cols = [c for c in syn_clusters.columns
             if c.startswith("count_") and c.endswith("le10km")]
if "cluster_id" in syn_clusters.columns and cand_cols:
    syn_clusters["n10_total"] = syn_clusters[cand_cols].sum(axis=1)
    top_cov = (syn_clusters
               .sort_values("n10_total", ascending=False)
               .head(10)[["cluster_id","NAM_2","n10_total"] + cand_cols])
    top_cov
else:
    print("Expected columns missing in syn_clusters (need cluster_id and count_*_le10km).")
```

**This cell flags sites with **zero projects within 30 km** (islands to treat with caution or to seed first).**

```{code-cell} ipython3
c30 = [c for c in syn_sites.columns
       if c.startswith("count_") and c.endswith("le30km")]
if c30:
    islands = syn_sites.loc[syn_sites[c30].sum(axis=1) == 0, ["site_id"] + c30]
    islands.head(20)
else:
    print("No count_*_le30km columns found in syn_sites.")
```

**This cell ranks municipalities by **density of nearby projects** around clusters (who can coordinate most).**

```{code-cell} ipython3
cand_cols = [c for c in syn_clusters.columns
             if c.startswith("count_") and c.endswith("le10km")]

if {"NAM_2","cluster_id"}.issubset(syn_clusters.columns) and cand_cols:
    muni_proj = (syn_clusters
                 .assign(n10_total=lambda d: d[cand_cols].sum(axis=1))
                 .groupby("NAM_2", as_index=False)
                 .agg(clusters=("cluster_id","count"),
                      n10_total=("n10_total","sum"))
                 .sort_values(["n10_total","clusters"], ascending=[False, False]))
    muni_proj.head(10)
else:
    print("NAM_2 or cluster_id missing, or no count_*_le10km columns in syn_clusters.")
```

**This cell draws a quick bar of Top-8 sites by “projects within 10 km (all partners)”.**

```{code-cell} ipython3
import matplotlib.pyplot as plt

c10 = [c for c in syn_sites.columns
       if c.startswith("count_") and c.endswith("le10km")]
if c10:
    syn_sites["n10_total"] = syn_sites[c10].sum(axis=1)
    t8 = syn_sites.nlargest(8, "n10_total")
    plt.figure()
    plt.barh(t8["site_id"].astype(str), t8["n10_total"])
    plt.gca().invert_yaxis()
    plt.xlabel("Projects within 10 km (Gov + WB + Other)")
    plt.title(f"{AOI}: Sites with strongest coordination potential")
    plt.show()
else:
    print("No count_*_le10km columns found in syn_sites.")
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

* Synergy **sites** → `outputs/tables/{AOI}_site_synergies.csv`
* Synergy **clusters** → `outputs/tables/{AOI}_cluster_synergies.csv`

# 2. Are we putting scarce resources where they matter most?

## Problem

Which **municipalities (Admin2)** should receive priority attention, and does that align with **poverty** and **food insecurity**?

## Strategy

Aggregate the priority surface to **Admin2** and build a **composite score** per municipality; then check equity by correlating with rural poverty and food-insecurity scale (RAPP).

## Data

- Priority rasters/tables from Chapter 1  
- Admin2 polygons + RAPP themes: **poverty**, **food insecurity**

## Methods (brief)

- **Step 09**: compute Admin2 KPIs (area, population, cropland km², % electrified, % rural, % area under the priority mask, access metrics) and a simple **min–max composite score**.
- Under the hood, the score averages normalized indicators such as **rural poverty**, **food insecurity**, **average travel time**, **share of priority area**, **cropland km²**, and **(1 − % electrified)** (see Step 09 docstring for details).
- Merge with RAPP **poverty** and **food insecurity** attributes.
- Compute correlations as a basic equity check (see Chapter 7 for detailed outlier flags at municipality level).


## Outputs

- `outputs/tables/{AOI}_priority_muni_rank.csv` — one row per Admin2 (composite score, population, and merged poverty & food-insecurity indicators)
- `outputs/tables/{AOI}_priority_scenarios_summary.csv` — stability across scenarios (optional)

## How to run (analyst)

Run: **06 → 07 → 09** (and **10** for scenarios). This chapter only **loads** saved outputs (no recomputation).


**This cell loads the municipality ranking table from `/outputs` and shows the top-10 by score.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT  = ROOT / "outputs" / "tables"

rank = pd.read_csv(OUT / f"{AOI}_priority_muni_rank.csv").sort_values("score", ascending=False)
rank[["NAM_1","NAM_2","score","pop_total"]].head(10)
```

## Quick results

This cell lists the Top-5 municipalities by composite score.

```{code-cell} ipython3
top5 = rank.head(5)[["NAM_1","NAM_2","score","pop_total"]].copy()
top5
```

This cell runs an equity check: correlation between score and a rural-poverty column (auto-detected if present).

```{code-cell} ipython3
pov_candidates = [
    "rural_poverty", "poverty_rural", "pov_rural",
    "RURAL_POV", "data1"  # use 'data1' only if your RAPP merge kept rural poverty under this name
]
pov_col = next((c for c in pov_candidates if c in rank.columns), None)

if pov_col:
    r_pov = rank["score"].corr(rank[pov_col])
    print(f"Correlation (score vs {pov_col}):", round(r_pov, 2))
else:
    print("No rural-poverty column found in rank; skipping correlation.")
```

## How to read the results

- **Top-ranked** Admin2 should either match field expectations or spark a review.
- **Equity lens:** positive correlation with rural poverty indicates we’re reaching poorer municipalities; negative or near-zero suggests a retune (Chapter 4).
- If equity correlation is weak, try the ‘**drop NTL/VEG**’ scenario or raise the **cropland threshold**; re-run Step 10 and refresh this page.

## Caveats

- Poverty/food-insecurity are **survey-modelled**; small-area noise is possible.
- Scores reflect **current weights**—scenario stability matters for confidence.

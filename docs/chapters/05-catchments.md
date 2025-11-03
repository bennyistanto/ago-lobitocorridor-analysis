# 5. Who benefits and how fast (catchments)

## Problem

How many **people** and how much **cropland** can reach key services or markets from project sites within **30/60/120 minutes**, and which sites deliver the most reach fastest?

## Strategy

Use the **minutes-to-market** surface as a friction raster, build **isochrone bands** (≤30, ≤60, ≤120) around each **project site**, and aggregate **population** and **cropland** within those bands. Compare sites by **reach per minute** and **marginal gains** between bands.

## Data

* **Travel time to market (1-km, minutes)** — Step 00
* **Project sites (points)** — Diversifica Mais / AOI sites
* **Population (WorldPop, 1-km, persons/pixel)** — Step 00
* **Cropland fraction (1-km, 0–1)** — Step 00
* *(Optional)* Rural mask & electrification (1-km) — Step 00

## Methods (brief)

* **Step 12** builds isochrone masks at **30/60/120 minutes** from each site using the minutes raster (no network routing needed).
* Aggregate **pop sum** and **cropland km²** per site & threshold.
* Compute **marginal reach** (e.g., 60 minus 30) and **efficiency** metrics (reach per minute).

## Outputs

* `outputs/tables/{AOI}_catchments_kpis.csv`
  *(columns: site\_id, thresh\_min, pop, cropland\_km2, rural\_pop (opt), electrified\_pop (opt), …)*
* `outputs/rasters/{AOI}_iso_{thresh}min.tif` *(optional, if you saved AOI-wide masks)*
* `outputs/rasters/{AOI}_site_{id}_iso_{thresh}min.tif` *(optional per-site masks)*

## How to run (analyst)

Run **Step 12** once. This chapter only **loads** saved outputs (no recomputation).

**This cell loads the catchment KPIs table for the current AOI and previews the first rows.**

```python
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT_T = ROOT / "outputs" / "tables"

catch = pd.read_csv(OUT_T / f"{AOI}_catchments_kpis.csv")
catch.head(10)
```

## Quick results

**This cell pivots totals by threshold (corridor-wide reach at 30/60/120 minutes).**

```python
totals = (catch.groupby("thresh_min", as_index=False)
                .agg(pop=("pop","sum"),
                     cropland_km2=("cropland_km2","sum")))
totals.sort_values("thresh_min")
```

**This cell identifies the Top-5 sites by people within 60 minutes (fast-reach leaders).**

```python
top60 = (catch.query("thresh_min == 60")
               .sort_values("pop", ascending=False)
               .head(5)[["site_id","pop","cropland_km2"]])
top60
```

**This cell computes marginal gains: (60 − 30) and (120 − 60) minutes per site (who benefits from extending the envelope).**

```python
wide = catch.pivot_table(index="site_id", columns="thresh_min", values=["pop","cropland_km2"], aggfunc="sum").fillna(0)
# columns like ('pop', 30), ('pop', 60), …
wide.columns = [f"{m}_{t}" for m,t in wide.columns]
wide["pop_gain_60_30"]  = wide.get("pop_60",0)  - wide.get("pop_30",0)
wide["pop_gain_120_60"] = wide.get("pop_120",0) - wide.get("pop_60",0)
wide["crop_gain_60_30"]  = wide.get("cropland_km2_60",0)  - wide.get("cropland_km2_30",0)
wide["crop_gain_120_60"] = wide.get("cropland_km2_120",0) - wide.get("cropland_km2_60",0)
wide.sort_values("pop_gain_60_30", ascending=False).head(10)
```

**This cell computes a simple efficiency metric: people per minute to 60 (pop\_60 / 60) and ranks sites.**

```python
eff = (catch.query("thresh_min == 60")[["site_id","pop","cropland_km2"]]
             .assign(pop_per_min=lambda d: d["pop"]/60.0,
                     crop_per_min=lambda d: d["cropland_km2"]/60.0)
             .sort_values("pop_per_min", ascending=False))
eff.head(10)
```

*(Optional tiny chart—safe to skip if you want a text-only book.)*
**This cell draws a quick bar of Top-8 sites by people ≤60 min.**

```python
import matplotlib.pyplot as plt

t8 = eff.head(8)
plt.figure()
plt.barh(t8["site_id"].astype(str), t8["pop"])
plt.gca().invert_yaxis()
plt.xlabel("People within 60 min (WorldPop)")
plt.title(f"{AOI}: Top sites by fast reach (≤60 min)")
plt.show()
```

## How to read the results (interpretation)

* **Fast-reach leaders:** Sites with the largest **pop ≤60 min** often deliver quick wins without major off-road upgrades.
* **Marginal gains:** Large **(60–30)** or **(120–60)** gains suggest value from modest road improvements or service extensions.
* **Cropland focus:** Where **cropland km²** grows sharply with time, logistics improvements can unlock production areas.
* **Balance:** Use this alongside priority clusters (Ch. 3) and municipality targets (Ch. 2) to choose sites that are both impactful and feasible.

## Caveats

* Minutes are **modelled** from a raster, not routed on a full road network; local speeds and seasonality can shift actual times.
* WorldPop and cropland are **proxies**; validate with local enumerations where available.
* If a site sits near a raster edge or nodata zone, catchment stats may undercount.

### Download

* Catchment **KPIs** → `outputs/tables/{AOI}_catchments_kpis.csv`
* Isochrone **masks** (optional) → `outputs/rasters/{AOI}_iso_{thresh}min.tif` or per-site variants if you saved them

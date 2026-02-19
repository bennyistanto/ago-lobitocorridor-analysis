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
# 5. Who benefits and how fast (catchments)

## Problem

How many **people** and how much **cropland** can reach key services or markets from project sites within **30/60/120 minutes**, and which sites deliver the most reach fastest?

## Strategy

Use the **minutes-to-market** surface as a friction raster, build **isochrone bands** (≤30, ≤60, ≤120) around each **project site**, and aggregate **population** and **cropland** within those bands. Compare sites by **reach per minute** and **marginal gains** between bands.

## Data

* **OSM roads (lines, WGS84)** — `PATHS.ROADS` (Step 00 / AOI prep)
* **Project sites (points, WGS84)** — `PATHS.SITES` (Diversifica Mais / AOI sites)
* **Target grid (1-km template)** — `PARAMS.TARGET_GRID` (same grid used for base rasters)
* **Population (WorldPop, 1-km, persons/pixel)** — `outputs/rasters/{AOI}_pop_1km.tif` (Step 00)
* **Cropland fraction (1-km, 0–1)** — `outputs/rasters/{AOI}_cropland_fraction_1km.tif` (Step 00)
* *(Optional)* **RWI (Relative Wealth Index, 1-km)** — if present in AOI rasters for additional equity KPIs

## Methods (brief)

* **Step 12** builds isochrone masks at **30/60/120 minutes** from each site using the minutes raster (no network routing needed).
* Aggregate **pop sum** and **cropland km²** per site & threshold.
* Compute **marginal reach** (e.g., 60 minus 30) and **efficiency** metrics (reach per minute).
* **Marginal catchment analysis**: for each site, compute the population reachable within the threshold from this site **but not currently served** by existing services. This answers: *"How many NET NEW beneficiaries does this investment unlock?"*

## Outputs

* `outputs/tables/{AOI}_catchments_kpis.csv`
  *(columns: site_index, lon, lat, thresh_min, area_km2, pop, cropland_km2, rwi_mean, rwi_pop_weighted, mean_travel_min)*

* `outputs/tables/{AOI}_marginal_catchment.csv`
  *(columns: site_index, lon, lat, thresh_min, pop_marginal, pop_total_catch, pct_marginal)*

* `outputs/rasters/{AOI}_catch_site{N}_{thresh}min.tif`
  *(binary isochrone masks per site & threshold; 1 = reached within thresh minutes)*

## How to run (analyst)

Run **Step 12** once. This chapter only **loads** saved outputs (no recomputation).

**This cell loads the catchment KPIs table for the current AOI and previews the first rows.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

# Go up two levels (../..) to get from /docs/chapters/ to the repo root
ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")
OUT_T = ROOT / "outputs" / "tables"

catch = pd.read_csv(OUT_T / f"{AOI}_catchments_kpis.csv")
catch.head(10)
```

## Quick results

**This cell pivots totals by threshold (corridor-wide reach at 30/60/120 minutes).**

```{code-cell} ipython3
totals = (catch.groupby("thresh_min", as_index=False)
                .agg(pop=("pop","sum"),
                     cropland_km2=("cropland_km2","sum")))
totals.sort_values("thresh_min")
```

**This cell identifies the Top-5 sites by people within 60 minutes (fast-reach leaders).**

```{code-cell} ipython3
top60 = (catch.query("thresh_min == 60")
               .sort_values("pop", ascending=False)
               .head(5)[["site_index","pop","cropland_km2"]])
top60
```

**This cell computes marginal gains: (60 − 30) and (120 − 60) minutes per site (who benefits from extending the envelope).**

```{code-cell} ipython3
wide = catch.pivot_table(index="site_index", columns="thresh_min",
                         values=["pop","cropland_km2"], aggfunc="sum").fillna(0)
# columns like ('pop', 30), ('pop', 60), …
wide.columns = [f"{m}_{t}" for m,t in wide.columns]
wide["pop_gain_60_30"]  = wide.get("pop_60",0)  - wide.get("pop_30",0)
wide["pop_gain_120_60"] = wide.get("pop_120",0) - wide.get("pop_60",0)
wide["crop_gain_60_30"]  = wide.get("cropland_km2_60",0)  - wide.get("cropland_km2_30",0)
wide["crop_gain_120_60"] = wide.get("cropland_km2_120",0) - wide.get("cropland_km2_60",0)
wide.sort_values("pop_gain_60_30", ascending=False).head(10)
```

**This cell computes a simple efficiency metric: people per minute to 60 (pop\_60 / 60) and ranks sites.**

```{code-cell} ipython3
eff = (catch.query("thresh_min == 60")[["site_index","pop","cropland_km2"]]
             .assign(pop_per_min=lambda d: d["pop"]/60.0,
                     crop_per_min=lambda d: d["cropland_km2"]/60.0)
             .sort_values("pop_per_min", ascending=False))

```

**This cell draws a quick bar of Top-8 sites by people ≤60 min.**

```{code-cell} ipython3
import matplotlib.pyplot as plt

t8 = eff.head(8)
plt.figure()
plt.barh(t8["site_index"].astype(str), t8["pop"])
plt.gca().invert_yaxis()
plt.xlabel("People within 60 min (WorldPop)")
plt.title(f"{AOI}: Top sites by fast reach (≤60 min)")
plt.show()
```

## Net new beneficiaries (marginal catchment)

The standard catchment analysis counts *everyone* within reach of a site, but many of those people may already have access through existing services. The **marginal catchment** shows only the people who gain access *for the first time* thanks to this investment.

**This cell loads the marginal catchment table (if available).**

```{code-cell} ipython3
marg_path = OUT_T / f"{AOI}_marginal_catchment.csv"
marg = pd.read_csv(marg_path) if marg_path.exists() else None

if marg is not None:
    print(f"Loaded marginal catchment data for {marg['site_index'].nunique()} sites")
    marg.head(15)
else:
    print("Marginal catchment table not found; run Step 12.")
```

**This cell ranks sites by net new beneficiaries at 60 minutes.**

```{code-cell} ipython3
if marg is not None:
    m60 = marg.query("thresh_min == 60").sort_values("pop_marginal", ascending=False)
    if not m60.empty:
        print("Top sites by NET NEW beneficiaries (60 min)")
        m60.head(10)[["site_index", "pop_marginal", "pop_total_catch", "pct_marginal"]]
```

**This cell compares total vs. marginal reach to highlight sites with the highest additionality.**

```{code-cell} ipython3
if marg is not None:
    m60 = marg.query("thresh_min == 60").copy()
    if not m60.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(m60["site_index"].astype(str), m60["pop_total_catch"], label="Total catchment", alpha=0.4)
        plt.bar(m60["site_index"].astype(str), m60["pop_marginal"], label="Net new (marginal)", alpha=0.8)
        plt.xlabel("Site")
        plt.ylabel("Population")
        plt.title(f"{AOI}: Total vs. marginal catchment at 60 min")
        plt.legend()
        plt.xticks(rotation=45)
        plt.show()
```

## How to read the results (interpretation)

* **Fast-reach leaders:** Sites with the largest **pop ≤60 min** often deliver quick wins without major off-road upgrades.
* **Net new beneficiaries:** Sites with high **pop_marginal** are unlocking access for people who currently have *no* service within that time band. These represent the highest additionality.
* **High pct_marginal:** A site where 80% of the catchment is marginal is serving a mostly unserved area. A site where only 10% is marginal is mostly duplicating existing coverage.
* **Marginal gains:** Large **(60–30)** or **(120–60)** gains suggest value from modest road improvements or service extensions.
* **Cropland focus:** Where **cropland km²** grows sharply with time, logistics improvements can unlock production areas.
* **Balance:** Use this alongside priority clusters (Ch. 3) and municipality targets (Ch. 2) to choose sites that are both impactful and feasible.
* **Equity lens:** Use `rwi_mean` and `rwi_pop_weighted` to see whether fast-reach sites mainly serve poorer or better-off areas (where RWI is available).

## Caveats

* Minutes are **modelled** from a raster, not routed on a full road network; local speeds and seasonality can shift actual times.
* WorldPop and cropland are **proxies**; validate with local enumerations where available.
* If a site sits near a raster edge or nodata zone, catchment stats may undercount.

### Download

* Catchment **KPIs** → `outputs/tables/{AOI}_catchments_kpis.csv`
* Marginal catchment → `outputs/tables/{AOI}_marginal_catchment.csv`
* Isochrone **masks** (optional) → `outputs/rasters/{AOI}_iso_{thresh}min.tif` or per-site variants if you saved them

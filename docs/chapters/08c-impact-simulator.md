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
# 8c. What if we upgrade a road?

## The decision question

Before committing budget to a road upgrade, decision makers want to know: **how many people would benefit, and by how much?** This chapter provides a simulation-based answer.

## What this analysis does

Step 16 takes the existing travel-time friction surface, applies a hypothetical road improvement (e.g., upgrading flood-vulnerable segments to secondary-road standard), and recomputes the travel-time surface from all project sites. The difference between *before* and *after* shows exactly where and how much travel time improves.

This turns an abstract infrastructure question into concrete numbers: X thousand people gain Y minutes of improved access.

## How it works (non-technical)

1. **Baseline**: Compute travel times from all project sites using today's road network
2. **Intervention**: Identify which road cells to upgrade (default: all flood-risk road cells from Step 04) and set them to a target speed (default: 45 km/h, secondary road standard)
3. **Recompute**: Calculate new travel times from the same project sites on the improved network
4. **Compare**: Subtract after from before to get minutes saved per grid cell
5. **Summarize**: Count how many people gain 5, 10, 15, 30, or 60+ minutes of improvement, and how many people newly fall within the 60-minute or 120-minute service threshold

## Data

* **Friction surface** — `outputs/rasters/{AOI}_friction_min_per_km.tif` (Step 12)
* **Population** — `outputs/rasters/{AOI}_pop_1km.tif` (Step 00)
* **Flood-risk road cells** — `outputs/rasters/{AOI}_roads_flood_risk_cells_1km.tif` (Step 04)
* **Project sites** — `PATHS.SITES` (Diversifica Mais candidate locations)

## Outputs

* `outputs/rasters/{AOI}_sim_travel_after.tif` — post-intervention travel surface
* `outputs/rasters/{AOI}_sim_travel_delta.tif` — minutes saved (positive = improvement)
* `outputs/tables/{AOI}_sim_impact_summary.csv` — population and area by improvement threshold

## How to run (analyst)

Run **Step 16** once. This chapter loads saved outputs.

**This cell loads the impact summary table.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")
OUT_T = ROOT / "outputs" / "tables"

impact_path = OUT_T / f"{AOI}_sim_impact_summary.csv"
impact = pd.read_csv(impact_path) if impact_path.exists() else None

if impact is not None:
    impact
else:
    print("Impact summary not found; run Step 16.")
```

## Quick results

**This cell shows the population gaining at least 10 minutes of improvement.**

```{code-cell} ipython3
if impact is not None:
    big_gains = impact[impact["min_saved_threshold"] == 10]
    if not big_gains.empty:
        pop = big_gains["pop_gaining"].iloc[0]
        area = big_gains["area_km2"].iloc[0]
        print(f"People gaining >= 10 min: {pop:,.0f}")
        print(f"Area improved:           {area:,.0f} km²")
```

**This cell shows how many people newly come within 60 and 120 minutes of a project site.**

```{code-cell} ipython3
if impact is not None:
    for iso in ("newly_within_60min", "newly_within_120min"):
        row = impact[impact["min_saved_threshold"] == iso]
        if not row.empty:
            print(f"{iso}: {row['pop_gaining'].iloc[0]:,.0f} people, {row['area_km2'].iloc[0]:,.0f} km²")
```

**This cell draws a bar chart of population gaining by improvement threshold.**

```{code-cell} ipython3
import matplotlib.pyplot as plt

if impact is not None:
    # Filter to numeric thresholds only
    numeric = impact[impact["min_saved_threshold"].apply(lambda x: str(x).isdigit())].copy()
    numeric["min_saved_threshold"] = numeric["min_saved_threshold"].astype(int)

    if not numeric.empty:
        plt.figure()
        plt.bar(numeric["min_saved_threshold"].astype(str), numeric["pop_gaining"])
        plt.xlabel("Minimum minutes saved")
        plt.ylabel("Population gaining (people)")
        plt.title(f"{AOI}: Who benefits from the road upgrade?")
        plt.show()
```

## How to read the results

* **Large populations at low thresholds** (e.g., many people gaining 5+ minutes) indicate widespread but modest improvements. This is typical of upgrading a few key segments on a busy route.
* **Smaller populations at high thresholds** (e.g., fewer people gaining 30+ minutes) indicate transformative improvements for remote communities. This often matters more for equity.
* **Newly within 60 min** is the single most important metric for decision makers: it counts people who *previously had no reasonable access* to project sites and now do.
* Compare across intervention scenarios by changing the target speed or selecting different road cells for upgrade.

## Customizing the scenario

The default scenario upgrades all flood-risk road cells to 45 km/h (secondary road standard). To test different scenarios:

* **Change target speed**: Modify the `target_speed_kmh` variable in `step_16_intervention_simulator.py`
* **Select different cells**: Replace the risk mask with your own selection (e.g., a specific road segment)
* **Multiple scenarios**: Run Step 16 multiple times with different settings and compare the impact tables

## Caveats

* The simulation uses a **cost-distance model**, not a full traffic assignment. It assumes all people travel to the nearest project site.
* **Road speeds are estimated** from OSM road classes, not measured. Actual improvements may differ.
* The default scenario upgrades **all** flood-risk cells simultaneously. In practice, improvements would be phased.
* For planning-grade analysis, pair this simulation with engineering cost estimates to compute cost-effectiveness ratios.

### Download

* Impact **summary** → `outputs/tables/{AOI}_sim_impact_summary.csv`
* Travel surface **after** → `outputs/rasters/{AOI}_sim_travel_after.tif`
* Travel time **delta** → `outputs/rasters/{AOI}_sim_travel_delta.tif`

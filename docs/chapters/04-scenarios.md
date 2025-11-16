# 4. What happens when we change the rules?

## Problem

Do our priorities **hold** if we tweak the rules (weights, masks, selection envelope), or do the “winners” flip when proxies like NTL/VEG are removed?

## Strategy

Predefine a small set of **scenarios** (e.g., “baseline”, “drop\_NTL\_VEG”, “high\_cropland”, “fixed\_km2”) and compare: (a) **municipality ranks**, (b) **cluster coverage**, and (c) **headline KPIs** side-by-side.

## Data

* **Scenario summary** — `outputs/tables/{AOI}_priority_scenarios_summary.csv` (Step 10; one row per scenario with weights, selection metrics, and overlaps vs baseline)
* *(Optional)* **Scenario masks** — `outputs/rasters/{AOI}_priority_mask_{scenario_id}.tif` (if masks are saved in Step 10)
* *(Optional)* **Baseline mask** — `outputs/rasters/{AOI}_priority_top10_mask.tif` (from Step 07; if missing, Step 10 treats the first scenario as the baseline in-memory)

## Methods (brief)

* **Step 10** defines a small list of **scenarios** (weights, component toggles, masks, selection envelope).
* For each scenario, it computes:
  * **selected_cells** and **selected_km2**
  * **pop_selected** and **ag_km2_selected** (cropland)
  * **drought_mean_selected** (if drought raster exists)
  * **overlap_pct_vs_baseline** and **jaccard_vs_baseline** vs a chosen baseline mask
* Results are written as a **scenario-level summary table** (one row per scenario). If you enable sidecar writing, a JSON meta file with the same scenario definitions is also saved.

## Outputs

* `outputs/tables/{AOI}_priority_scenarios_summary.csv`  
  *(columns: scenario_id, desc, weight & mask parameters, selected_cells, selected_km2, pop_selected, ag_km2_selected, drought_mean_selected, overlap_pct_vs_baseline, jaccard_vs_baseline, …)*
* *(Optional)* `outputs/rasters/{AOI}_priority_mask_{scenario_id}.tif` — per-scenario Top-X masks
* *(Optional)* `outputs/tables/{AOI}_priority_muni_rank.csv` — baseline municipality ranking from Step 09 (used in Chapters 2 and 7)

## How to run (analyst)

Run **Step 10** once to generate the scenario summary (and optional masks/meta).  
This chapter only **loads** the saved summary (no recomputation).

**This cell loads the scenario summary and the baseline rank for the current AOI.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT  = ROOT / "outputs" / "tables"

summary_path = OUT / f"{AOI}_priority_scenarios_summary.csv"
rank_path    = OUT / f"{AOI}_priority_muni_rank.csv"

summary = pd.read_csv(summary_path) if summary_path.exists() else None
rank    = pd.read_csv(rank_path).sort_values("score", ascending=False) if rank_path.exists() else None

print("Loaded:", summary_path.name if summary is not None else "no summary",
      "|", rank_path.name if rank is not None else "no baseline")
```

## Quick results

**This cell loads the scenario summary for the current AOI.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT  = ROOT / "outputs" / "tables"

summary_path = OUT / f"{AOI}_priority_scenarios_summary.csv"
summary = pd.read_csv(summary_path) if summary_path.exists() else None

summary.head(10) if summary is not None else f"Summary not found at: {summary_path}"
```

**This cell lists the scenarios with their main parameters (weights, masks).**

```{code-cell} ipython3
if summary is not None:
    cols_basic = ["scenario_id", "desc"]
    # keep any weight / mask columns if present
    extra = [c for c in summary.columns if c.startswith(("W_", "MASK_", "TOP_", "MIN_CLUSTER_"))]
    summary[cols_basic + extra].drop_duplicates("scenario_id")
else:
    print("Run Step 10 first to generate the summary.")
```

**This cell compares key coverage metrics across scenarios.**

```{code-cell} ipython3
if summary is not None:
    keep = ["scenario_id", "desc",
            "selected_km2", "pop_selected",
            "ag_km2_selected", "drought_mean_selected",
            "overlap_pct_vs_baseline", "jaccard_vs_baseline"]
    [c for c in keep if c in summary.columns]  # just a quick check
    summary[keep].sort_values("pop_selected", ascending=False)
```

**This cell shows which scenarios cover the largest area and population.**

```{code-cell} ipython3
if summary is not None:
    top_pop = summary.sort_values("pop_selected", ascending=False).head(5)
    top_area = summary.sort_values("selected_km2", ascending=False).head(5)
    top_pop[["scenario_id","desc","pop_selected","selected_km2"]], \
    top_area[["scenario_id","desc","selected_km2","pop_selected"]]
```

**This cell orders scenarios by overlap vs. the baseline (stability).**

```{code-cell} ipython3
if summary is not None and "overlap_pct_vs_baseline" in summary.columns:
    stab = (summary[["scenario_id","desc","overlap_pct_vs_baseline","jaccard_vs_baseline"]]
                  .sort_values("overlap_pct_vs_baseline", ascending=False))
    stab
else:
    print("Overlap columns not found; check Step 10 output.")
```

**This cell draws a bar chart of population covered by each scenario.**

```{code-cell} ipython3
import matplotlib.pyplot as plt

if summary is not None:
    plt.figure()
    (summary.sort_values("pop_selected", ascending=False)
            .set_index("scenario_id")["pop_selected"]
            .plot(kind="bar"))
    plt.ylabel("Population in selected mask")
    plt.title(f"{AOI}: Scenario comparison by population coverage")
    plt.tight_layout()
    plt.show()
```

## How to read the results (interpretation)

* **Robust choices** = appear in **Top-K across all scenarios** → safe early picks.
* **Swing choices** = large **rank deltas** → investigate why (often NTL/VEG or cropland threshold effects).
* **Explainability** = when decision changes after a single tweak, surface that assumption explicitly in slides/notes.

## Caveats

* If a scenario drops a proxy (e.g., NTL/VEG), lower correlation with poverty may actually be **good** (less bias).
* Comparing **Top-% vs. Top-km²** changes the selection envelope; treat results separately when reporting.

### Download

* Scenario **summary** → `outputs/tables/{AOI}_priority_scenarios_summary.csv`
* Scenario **meta** (if enabled) → `outputs/tables/{AOI}_priority_scenarios.meta.json`
* Baseline **muni rank** for equity checks → `outputs/tables/{AOI}_priority_muni_rank.csv`

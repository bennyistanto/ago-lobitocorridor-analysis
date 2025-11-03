# 4. What happens when we change the rules?

## Problem

Do our priorities **hold** if we tweak the rules (weights, masks, selection envelope), or do the “winners” flip when proxies like NTL/VEG are removed?

## Strategy

Predefine a small set of **scenarios** (e.g., “baseline”, “drop\_NTL\_VEG”, “high\_cropland”, “fixed\_km2”) and compare: (a) **municipality ranks**, (b) **cluster coverage**, and (c) **headline KPIs** side-by-side.

## Data

* **Scenario run tables** — Step 09 (per-scenario ranks & metrics)
* **Scenario summary** — Step 10 (stacked comparison across scenarios)
* **Priority clusters** — Step 11 (for optional overlap checks)

## Methods (brief)

* **Step 09**: run multiple **parameter bundles** (weights/masks/selection).
* **Step 10**: consolidate into tidy comparison tables (ranks, score deltas, cluster/coverage deltas).
* We report **stability** (how often a municipality stays in top-K) and **swing** (rank change).

## Outputs

* `outputs/tables/{AOI}_priority_scenarios_summary.csv` — municipality-level comparison across scenarios (rank, score, top-K flags)
* `outputs/tables/{AOI}_priority_scenarios_clusters.csv` *(optional)* — cluster coverage/metrics per scenario
* `outputs/tables/{AOI}_priority_muni_rank.csv` — baseline for reference

## How to run (analyst)

Run **Step 09 → Step 10** once. This chapter only **loads** saved outputs (no recomputation).

**This cell loads the scenario summary and the baseline rank for the current AOI.**

```python
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

**This cell shows the list of scenarios available in the summary.**

```python
if summary is None:
    print("Scenario summary not found; run Steps 09–10 first.")
else:
    scenarios = sorted(summary["scenario"].unique().tolist())
    print("Scenarios:", scenarios)
```

**This cell builds a Top-5 cross-scenario stability table (how often each municipality appears in Top-5).**

```python
if summary is not None:
    # Expect columns: NAM_2, scenario, rank (1=best)
    top5 = summary[summary["rank"] <= 5]
    stab = (top5.groupby("NAM_2", as_index=False)["scenario"]
                 .nunique()
                 .rename(columns={"scenario":"appearances_in_top5"}))
    stab = stab.sort_values("appearances_in_top5", ascending=False)
    stab.head(15)
```

**This cell compares ranks between two chosen scenarios (delta = scenario\_b − scenario\_a).**

```python
# Choose two scenarios by name; adjust to your set
scenario_a = "baseline"
scenario_b = "drop_NTL_VEG"

if summary is not None and all(s in summary["scenario"].unique() for s in [scenario_a, scenario_b]):
    a = summary[summary["scenario"] == scenario_a][["NAM_2","rank"]].rename(columns={"rank":"rank_a"})
    b = summary[summary["scenario"] == scenario_b][["NAM_2","rank"]].rename(columns={"rank":"rank_b"})
    comp = a.merge(b, on="NAM_2", how="inner")
    comp["rank_delta"] = comp["rank_b"] - comp["rank_a"]  # negative = improved
    comp.sort_values("rank_delta").head(15)
else:
    print("Adjust scenario_a/scenario_b to match your summary.")
```

**This cell flags municipalities that are consistently Top-K across all scenarios (robust picks).**

```python
TOP_K = 5

if summary is not None:
    scen_count = summary["scenario"].nunique()
    robust = (summary.query("rank <= @TOP_K")
                     .groupby("NAM_2", as_index=False)["scenario"].nunique()
                     .query("scenario == @scen_count")
                     .sort_values("NAM_2"))
    print(f"Municipalities in Top-{TOP_K} for ALL {scen_count} scenarios:")
    robust["NAM_2"].tolist()
```

**This cell lists the biggest ‘swing’ municipalities (largest absolute rank change across any two scenarios).**

```python
if summary is not None:
    piv = summary.pivot_table(index="NAM_2", columns="scenario", values="rank")
    piv["max_swing"] = (piv.max(axis=1) - piv.min(axis=1)).astype("int64")
    piv.sort_values("max_swing", ascending=False).head(15)[["max_swing"]]
```

*(Optional tiny chart—safe to skip if you want a text-only book.)*
**This cell draws a quick bar of Top-10 max swings to visualize sensitivity.**

```python
import matplotlib.pyplot as plt

if summary is not None:
    swings = (summary.pivot_table(index="NAM_2", columns="scenario", values="rank")
                     .assign(max_swing=lambda d: d.max(axis=1)-d.min(axis=1))
                     .sort_values("max_swing", ascending=False)
                     .head(10)["max_swing"])
    plt.figure()
    swings.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.xlabel("Rank swing (bigger = more sensitive)")
    plt.title(f"{AOI}: Top-10 municipalities by scenario sensitivity")
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
* Scenario **clusters** (optional) → `outputs/tables/{AOI}_priority_scenarios_clusters.csv`
* Baseline **muni rank** → `outputs/tables/{AOI}_priority_muni_rank.csv`

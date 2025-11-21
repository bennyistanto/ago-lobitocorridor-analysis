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
# 10. Run this anywhere along the corridor (AOI playbook)

## Problem

We need a **repeatable recipe** to rerun the analysis for **any AOI/province** along the Lobito Corridor without breaking paths, CRS, or filenames.

## Strategy

Standardize three things: (1) **AOI selection** (single source of truth), (2) **folder structure** (`data/`, `outputs/`), and (3) **script order**. Provide quick **preflight checks** for inputs and a **minimal run list** that writes everything needed for the book.

## Data

* Same inputs as earlier chapters (see “Data menu”) but **clipped per AOI**.
* Socio-economic Admin2 (RAPP) shapefiles:  
  `data/vectors/{pfx}_{AOI}_adm2_{theme}_rapp_2020_a.shp`,  
  where `{pfx} ∈ {ago_gov, ago_pop}` and `theme` is one of the RAPP themes (e.g. `poverty`, `traveltime`, `foodinsecurity`).
* Core rasters/vectors following your standardized names in `config.py`.

## Methods (brief)

* Pick AOI via `AOI` env var (or notebook variable) → `config.py` builds paths using `{AOI}`.
* Run **core steps** in order (00→…); most chapters only **load** results from `outputs/`.

## Outputs

* All rasters under `outputs/rasters/` and tables under `outputs/tables/`, prefixed with `{AOI}_…`.
* Chapters 1–9 auto-refresh by **loading** these outputs (no recomputation in docs).

---

## How to run (analyst)

**This cell sets/reads AOI & project root; it prints where outputs will go.**

```{code-cell} ipython3
import os
from pathlib import Path

# Go up two levels (../..) to get from /docs/chapters/ to the repo root
ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")
OUT_R = ROOT / "outputs" / "rasters"
OUT_T = ROOT / "outputs" / "tables"
print("AOI:", AOI, "\nROOT:", ROOT, "\nOUT_R:", OUT_R, "\nOUT_T:", OUT_T)
```

**This cell imports `config.py` to confirm paths resolve for this AOI.**

```{code-cell} ipython3
import sys
sys.path.append(str(ROOT / "src"))
from config import PATHS, PARAMS

print("Key inputs for AOI:")
print("  TRAVEL:", PATHS.TRAVEL.name)
print("  POP   :", PATHS.POP.name)
print("  CROPL :", PATHS.CROPLAND.name)
print("  ELEC  :", PATHS.ELEC.name)
print("  SETTLE:", PATHS.SETTLE.name)
print("  FLOOD :", PATHS.FLOOD.name)
```

**This cell runs a quick preflight: do required files exist for this AOI?**

```{code-cell} ipython3
missing = [k for k,p in {
    "TRAVEL": PATHS.TRAVEL, "POP": PATHS.POP, "NTL": PATHS.NTL, "VEG": PATHS.VEG,
    "DROUGHT": PATHS.DROUGHT, "FLOOD": PATHS.FLOOD, "CROPLAND": PATHS.CROPLAND,
    "ELEC": PATHS.ELEC, "SETTLE": PATHS.SETTLE, "BND_ADM1": PATHS.BND_ADM1, "SITES": PATHS.SITES
}.items() if not p.exists()]
print("Missing:", missing if missing else "All required inputs found.")
```

**This cell shows the *socio-economic Admin2 themes* available (from config).**

```{code-cell} ipython3
from config import ADMIN2_THEMES
ADMIN2_THEMES
```

**This cell prints the *minimal core run list* (scripts → outputs they create).**

```{code-cell} ipython3
print(
    "Run order (minimal):\n"
    "  00  Align & rasterize core layers   → outputs/rasters/{AOI}_*_1km.tif\n"
    "  07  Priority surface + Admin2 rank  → outputs/tables/{AOI}_priority_admin2_rank.csv\n"
    "  09  Municipality targeting table    → outputs/tables/{AOI}_priority_muni_rank.csv\n"
    "  10  (Optional) Scenario summary     → outputs/tables/{AOI}_priority_scenarios_summary.csv\n"
    "  11  Priority clusters + KPIs        → outputs/tables/{AOI}_priority_clusters.csv\n"
    "  12  Catchments from sites           → outputs/tables/{AOI}_catchments_kpis.csv\n"
    "  13  Synergies (project proximity)   → outputs/tables/{AOI}_site_synergies.csv, {AOI}_cluster_synergies.csv\n"
    "  14  Origin Destination (optional)   → outputs/tables/{AOI}_od_gravity.csv, {AOI}_od_zone_attrs.csv, {AOI}_od_agents.csv\n"
)
```

**This cell (optional) shows how to run a step module inline from the book (use sparingly).**

```{code-cell} ipython3
# ⚠️ Prefer running steps from your notebooks/CLI; chapters should mostly load outputs.
# Example: run step_07 in-process if you must refresh quickly.
import importlib
m = importlib.import_module("step_07_priority_tunable")
m.main()  # writes tables into outputs/
```

---

## Quick results

**This cell verifies that the *minimum expected outputs* exist after the run.**

```{code-cell} ipython3
expected = [
    OUT_T / f"{AOI}_priority_muni_rank.csv",
    OUT_T / f"{AOI}_priority_clusters.csv",
    OUT_T / f"{AOI}_catchments_kpis.csv"
]
[(p.name, p.exists()) for p in expected]
```

**This cell previews the top lines of the municipality ranking (sanity check).**

```{code-cell} ipython3
import pandas as pd
rank_path = OUT_T / f"{AOI}_priority_muni_rank.csv"
pd.read_csv(rank_path).head(5) if rank_path.exists() else "Run Step 07 first."
```

**This cell previews the top lines of the cluster KPIs (sanity check).**

```{code-cell} ipython3
clu_path = OUT_T / f"{AOI}_priority_clusters.csv"
pd.read_csv(clu_path).head(5) if clu_path.exists() else "Run Step 11 first."
```

---

## How to read the results (interpretation)

* If the **sanity checks** above load cleanly, the rest of the book pages will render for the new AOI.
* If files are missing, re-run the step(s) that produce them (see run list).
* Keep the **same parameter block** across AOIs if you want apples-to-apples comparisons (see Chapter 9).

## Caveats

* Ensure the AOI’s inputs follow the same **naming conventions** expected by `config.py`.
* If you switch AOI frequently, consider **clearing stale rasters** under `outputs/rasters/` for that AOI to avoid mixing old/new grids.
* Road tagging in OSM can vary by province; travel-time rasters will inherit these differences.

### Download

Nothing to download here—the playbook simply helps you trigger/verify the pipeline for a new AOI. Use the chapter-specific downloads (Ch. 1–8) once outputs exist.

# 15. Appendix (for analysts)

## Problem

Analysts need a **single reference** that maps scripts → parameters → outputs, and quick helpers to **inspect inventories** without diving into code.

## Strategy

Provide a **crosswalk** of steps (00→14), a **parameter glossary** (what each knob does), and tiny **inspection cells** for `/outputs`.

## Data

No new data: this page **reads** `config.py` and lists files under `outputs/`.

## Methods (brief)

* Show **step responsibilities** and the **exact artifacts** they write.
* Surface the **full PARAMS** block (not just the subset).
* Provide small utilities to **inventory** outputs (tables/rasters) by AOI.

## Outputs

None—reference page only.

---

## Step crosswalk (scripts → outputs)

| Step | Script (src/)                                   | What it does                                                                | Key outputs (outputs/)                                                                                                                                                                                                                                                                                                  |
| ---- | ----------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 00   | `step_00_align_and_rasterize.py`                | Aligns rasters to 1-km grid; rasterizes vectors; aggregates flood 30 m→1 km | `rasters/{AOI}_pop_1km.tif`, `{AOI}_ntl_1km.tif`, `{AOI}_veg_1km.tif`, `{AOI}_drought_1km.tif`, `{AOI}_cropland_presence_1km.tif`, `{AOI}_cropland_fraction_1km.tif`, `{AOI}_elec_grid_1km.tif`, `{AOI}_elec_unelectrified_1km.tif`, `{AOI}_urban_1km.tif`, `{AOI}_rural_1km.tif`, `{AOI}_flood_rp100_maxdepth_1km.tif` |
| 01   | `step_01_isochrones.py`                         | Builds AOI-wide 30/60/120-min masks                                         | `rasters/{AOI}_iso_30min.tif`, `…60…`, `…120…`                                                                                                                                                                                                                                                                          |
| 02   | `step_02_kpis_population_cropland_electric.py`  | Zonal stats for isochrones and basics                                       | `tables/{AOI}_isochrones_kpis.csv`                                                                                                                                                                                                                                                                                      |
| 03   | `step_03_priority_surface.py`                   | (Legacy) baseline priority surface                                          | `rasters/{AOI}_priority_legacy.tif` (optional)                                                                                                                                                                                                                                                                          |
| 04   | `step_04_flood_bottlenecks_from_road_raster.py` | Simple flood bottleneck screen (priority × flood)                           | `rasters/{AOI}_priority_flood_screen_1km.tif`                                                                                                                                                                                                                                                                           |
| 05   | `step_05_site_audit_points.py`                  | Samples rasters at project sites + 3×3 ring stats                           | `tables/{AOI}_site_audit_13_points.csv`                                                                                                                                                                                                                                                                                 |
| 06   | `step_06_muni_ingest.py`                        | Ingests Admin2 RAPP shapefiles (themes)                                     | `tables/{AOI}_admin2_themes_catalog.csv`                                                                                                                                                                                                                                                                                |
| 07   | `step_07_priority_tunable.py`                   | Priority score (tunable) + muni rank                                        | `tables/{AOI}_priority_muni_rank.csv`, `rasters/{AOI}_priority_score_1km.tif`                                                                                                                                                                                                                                           |
| 08   | `step_08_project_kpis.py`                       | Site KPls near priority & access                                            | `tables/{AOI}_project_kpis.csv`                                                                                                                                                                                                                                                                                         |
| 09   | `step_09_scenario_sweep.py`                     | Runs named parameter bundles                                                | scenario-wise tables in `tables/…`                                                                                                                                                                                                                                                                                      |
| 10   | `step_10_priority_scenarios.py`                 | Consolidates scenarios                                                      | `tables/{AOI}_priority_scenarios_summary.csv`                                                                                                                                                                                                                                                                           |
| 11   | `step_11_priority_clusters.py`                  | Labels Top-X, prunes, computes cluster KPIs                                 | `rasters/{AOI}_priority_top10_mask.tif` (or km² name), `rasters/{AOI}_priority_clusters_1km.tif`, `tables/{AOI}_priority_clusters.csv`                                                                                                                                                                                  |
| 12   | `step_12_catchments.py`                         | Catchments per site (≤30/60/120 min)                                        | `tables/{AOI}_catchments_kpis.csv`                                                                                                                                                                                                                                                                                      |
| 13   | `step_13_synergies.py`                          | Proximity to Gov/WB/Other projects                                          | `tables/{AOI}_synergy_sites.csv`, `tables/{AOI}_synergy_clusters.csv`                                                                                                                                                                                                                                                   |
| 14   | `step_14_od_lite.py`                            | Admin2 gravity flows + agent samples                                        | `tables/{AOI}_od_flows.csv`, `tables/{AOI}_od_agents_sample.csv`                                                                                                                                                                                                                                                        |

> Exact filenames may vary slightly if you selected **Top-km²** instead of **Top-%** in Step 10.

---

## Parameter glossary (what each knob does)

* **Weights:** `W_POP`, `W_NTL`, `W_VEG`, `W_DRT` — contribution of each normalized layer to the priority score.
* **Masks:** `MASK_MIN_CROPLAND` (min 0–1 fraction), `MASK_URBAN_EXCLUDE` (bool), `MASK_ELEC_EXISTING` (bool).
* **Selection:** `TOP_PCT_CELLS` (0–100, percent of cells) **or** `TOP_KM2` (fixed area).
* **Coherence & smoothing:** `MIN_CLUSTER_CELLS`, `MIN_CLUSTER_KM2`, `GAUSS_SIGMA_CELLS`.
* **Road class:** `ROAD_CLASS_FILTER` (“ALL”, or list like `["motorway","trunk","primary","secondary"]`) — used where steps consume road-based context.
* **Admin2 themes:** `ADMIN2_THEMES`, `THEME_VARS` — what socio-economic themes/columns are available.

---

## How to run (analyst)

**This cell shows the full `PARAMS` dict as currently loaded (read-only).**

```python
import os, sys, pprint
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
sys.path.append(str(ROOT / "src"))

from config import PARAMS
pp = pprint.PrettyPrinter(width=110, compact=True)
print("AOI:", AOI)
pp.pprint(PARAMS)
```

**This cell inventories output tables for the AOI (name, size, mtime).**

```python
import time
from pathlib import Path

OUT_T = ROOT / "outputs" / "tables"
rows = []
for p in sorted(OUT_T.glob(f"{AOI}_*.csv")):
    st = p.stat()
    rows.append({"file": p.name, "size_kb": round(st.st_size/1024,1),
                 "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
rows[:25]  # show first 25
```

**This cell inventories output rasters for the AOI (name, size, mtime).**

```python
OUT_R = ROOT / "outputs" / "rasters"
rows_r = []
for p in sorted(OUT_R.glob(f"{AOI}_*.tif")):
    st = p.stat()
    rows_r.append({"file": p.name, "size_mb": round(st.st_size/1024/1024,2),
                   "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime))})
rows_r[:25]
```

**This cell previews the top rows of the key decision tables (if present).**

```python
import pandas as pd

def _preview(path):
    return pd.read_csv(path).head(5) if path.exists() else f"Missing: {path.name}"

rank_path = OUT_T / f"{AOI}_priority_muni_rank.csv"
clu_path  = OUT_T / f"{AOI}_priority_clusters.csv"
cat_path  = OUT_T / f"{AOI}_catchments_kpis.csv"

_preview(rank_path), _preview(clu_path), _preview(cat_path)
```

---

## How to read the results (interpretation)

* The **crosswalk** tells you *which* step to re-run when a table is missing or stale.
* The **glossary** reminds you which knob to change for the behavior you want, without editing code.
* The **inventories** help keep notebooks tidy—if files look old or too small, revisit the producing step.

## Caveats

* Keep AOI-specific artifacts separate (your `{AOI}_` prefix already does this).
* If you alter the grid or CRS, **clear old rasters** to avoid mixing transforms.
* Scenario tables can grow quickly; prune unused runs to keep diffs readable.

### Download

Nothing to download—this page is a reference. Use earlier chapters to export decision tables and maps.

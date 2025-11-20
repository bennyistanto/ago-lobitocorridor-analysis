# 13. Limits & “handle with care”

## Problem

Public/EO data are powerful for triage—but some signals can **mislead** if read literally. This page lists key **failure modes**, what to **validate**, and quick **sanity checks** before decisions.

## Strategy

Document the main **limitations** by layer and method (resolution, bias, seasonality, alignment, model assumptions). Provide **simple diagnostics** to reveal common pitfalls (missing files, mismatched grids, extreme values).

## Data

No new data—this page only **inspects** rasters/tables already in `/outputs` and paths declared in `config.py`.

## Methods (brief)

* Spot **grid/CRS mismatches** across 1-km rasters.
* Check **value ranges** (e.g., minutes, drought %, cropland fraction).
* Flag **suspicious sparsity** (e.g., zero pop or cropland over large areas unexpectedly).
* Remind users where to **ground-truth** (field teams, PIUs, contractors).

## Outputs

None—this is a reading and diagnostics page.

---

## How to run (analyst)

**This cell loads config, sets paths, and prints the AOI.**

```{code-cell} ipython3
import os, sys
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "huambo")
sys.path.append(str(ROOT / "src"))

from config import PATHS

print("AOI:", AOI)
print("ROOT:", ROOT)
```

---

## Common limits (read first)

**Data & proxies**

* **WorldPop (1-km):** modelled; can under/over-estimate in sparsely enumerated rural areas. Treat as **relative density**, not exact counts.
* **Travel time (minutes raster):** depends on speed/friction assumptions and OSM coverage (unpaved, seasonal roads). Minutes are **screening**, not routing.
* **Night-time lights (NTL):** urban-biased proxy; can overweight peri-urban edges and miss rural activity.
* **Vegetation indices (VEG):** seasonality & cloud gaps; “green ≠ crops.” Use **cropland fraction** to focus.
* **Drought/flood layers:** different baselines & return periods; **do not mix** units or periods casually.

**Methods**

* **Priority score:** sum of **normalized** layers—interpret as **relative opportunity**, not probability.
* **Top-% vs Top-km²:** different envelopes; report clearly which you used.
* **Clusters (1-km):** schematic blobs; **not** project footprints.
* **Catchments:** **isochrone over raster**, not road network routing.
* **OD-Lite:** gravity model shows **interaction potential**, not observed trips.

**Practical**

* **CRS / grid alignment:** any stale raster with a different transform can corrupt downstream maps—clear old outputs if you change AOI or grid.
* **Season/time:** make sure years align (e.g., 2024 NTL vs 2020 drought baseline).

---

## Quick diagnostics

**This cell checks that key derived rasters exist and share the same grid (shape & transform).**

```{code-cell} ipython3
import rasterio as rio

OUT_R = ROOT / "outputs" / "rasters"
cands = [
    OUT_R / f"{AOI}_pop_1km.tif",
    OUT_R / f"{AOI}_veg_1km.tif",
    OUT_R / f"{AOI}_ntl_1km.tif",
    OUT_R / f"{AOI}_drought_1km.tif",
    OUT_R / f"{AOI}_cropland_fraction_1km.tif",
]

infos = []
for p in cands:
    if p.exists():
        with rio.open(p) as ds:
            infos.append((p.name, ds.width, ds.height, ds.transform, str(ds.crs)))
    else:
        infos.append((p.name, "MISSING", "", "", ""))
infos
```

**This cell flags mismatched shapes/transforms (a common source of odd maps).**

```{code-cell} ipython3
shapes = {(w,h) for (_,w,h,_,_) in infos if isinstance(w,int)}
transforms = {t for (_,w,h,t,_) in infos if hasattr(t,"a")}
print("Distinct shapes:", shapes)
print("Distinct transforms:", len(transforms))
```

**This cell checks basic value ranges for priority-related inputs (sanity only).**

```{code-cell} ipython3
import numpy as np
import rasterio as rio

def _range(path):
    if not path.exists(): return (path.name, "MISSING")
    with rio.open(path) as ds:
        arr = ds.read(1, masked=True)
        return (path.name, float(np.nanmin(arr)), float(np.nanmax(arr)))

checks = [
    OUT_R / f"{AOI}_pop_1km.tif",
    OUT_R / f"{AOI}_cropland_fraction_1km.tif",  # expect 0..1
    OUT_R / f"{AOI}_drought_1km.tif",            # expect ~0..100 (%) if that scale was used
    OUT_R / f"{AOI}_ntl_1km.tif",                 # 0..1 (if you normalized to 0..1)
]
[_range(p) for p in checks]
```

---

## How to read the results (interpretation)

* **Grid mismatch:** If shapes/transform differ → re-run **Step 00** and **clear stale outputs** for this AOI.
* **Range anomalies:** Cropland fraction outside **0..1**, or drought >100, suggests mixing scales; revisit **Step 00** harmonization.
* **Sparse signals:** If large swaths show zero pop/cropland where you expect activity, cross-check raw inputs and consider **raising weights** on more reliable layers.

## Caveats

* Diagnostics here are **lightweight**; they won’t catch subtle modelling issues.
* Local knowledge should always **override proxy weirdness**—use these tools to focus conversations, not to replace fieldwork.
* If you change AOI repeatedly, ensure `config.py` switches filenames **everywhere** (see Chapter 10) and purge AOI-specific outputs.

### Download

Nothing to download—this is a **checklist**. Use earlier chapters to download analysis tables and maps.

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
# 11. What data went in (data menu)

## Problem

Stakeholders need to know **what layers** power the analysis—sources, units, resolution—and verify that files exist for the current **AOI**.

## Strategy

Show a compact, machine-readable **catalog** of inputs and key derived rasters, plus the **Admin2 RAPP** themes at hand. Keep units/resolution explicit and highlight missing files.

## Data

This page **inspects** files referenced by `config.py` and those written by Step 00; it does not recompute anything.

## Methods (brief)

* Read `config.PATHS` and list canonical **input rasters/vectors** for the AOI.
* Stat key **aligned 1-km** rasters from Step 00 (pop/ntl/veg/drought, cropland presence & fraction, settlement/electricity masks, flood maxdepth 1-km).
* Enumerate **RAPP Admin2 themes** defined in `config.ADMIN2_THEMES` / `THEME_VARS`.
* (Optional) RWI grid (Meta, -2..+2) aligned at 1-km from Step 00, used for equity-sensitive overlays and OD mass tilt.

## Outputs

None—this is a **catalog** view only. (If you want, export the table at the bottom.)

---

## How to run (analyst)

**This cell loads config and prints key paths for the current AOI.**

```{code-cell} ipython3
import os, sys
from pathlib import Path

# Go up two levels (../..) to get from /docs/chapters/ to the repo root
ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")
sys.path.append(str(ROOT / "src"))

from config import PATHS, ADMIN2_THEMES, THEME_VARS

print("AOI:", AOI)
print("Data root:", ROOT / "data")
print("Outputs:", ROOT / "outputs")
print("Travel:", PATHS.TRAVEL.name)
print("Pop   :", PATHS.POP.name)
print("NTL   :", PATHS.NTL.name)
print("VEG   :", PATHS.VEG.name)
print("Drought:", PATHS.DROUGHT.name)
print("Flood :", PATHS.FLOOD.name)
print("Cropland:", PATHS.CROPLAND.name)
print("Electr:", PATHS.ELEC.name)
print("Settle:", PATHS.SETTLE.name)
```

**This cell compiles a table of core input files and whether they exist.**

```{code-cell} ipython3
import pandas as pd

core = {
    "travel_time_min_1km_input": PATHS.TRAVEL,
    "population_1km_input": PATHS.POP,
    "ntl_300m_input": PATHS.NTL,
    "vegindex_1km_input": PATHS.VEG,
    "drought_1km_input": PATHS.DROUGHT,
    "flood_depth_30m_input": PATHS.FLOOD,
    "cropland_10m_vector": PATHS.CROPLAND,
    "electricity_vector": PATHS.ELEC,
    "settlement_vector": PATHS.SETTLE,
    "adm1_boundary": PATHS.BND_ADM1,
    "project_sites": PATHS.SITES,
}
df_core = pd.DataFrame(
    [{"key": k, "path": str(p), "exists": p.exists()} for k, p in core.items()]
).sort_values("key")
df_core
```

**This cell stats the aligned 1-km rasters from Step 00 (if present).**

```{code-cell} ipython3
import rasterio as rio

OUT_R = ROOT / "outputs" / "rasters"
derived_candidates = {
    "pop_1km": OUT_R / f"{AOI}_pop_1km.tif",
    "ntl_1km": OUT_R / f"{AOI}_ntl_1km.tif",
    "veg_1km": OUT_R / f"{AOI}_veg_1km.tif",
    "drought_1km": OUT_R / f"{AOI}_drought_1km.tif",
    "cropland_presence_1km": OUT_R / f"{AOI}_cropland_presence_1km.tif",
    "cropland_fraction_1km": OUT_R / f"{AOI}_cropland_fraction_1km.tif",
    "elec_grid_1km": OUT_R / f"{AOI}_elec_grid_1km.tif",
    "elec_unelectrified_1km": OUT_R / f"{AOI}_elec_unelectrified_1km.tif",
    "urban_1km": OUT_R / f"{AOI}_urban_1km.tif",
    "rural_1km": OUT_R / f"{AOI}_rural_1km.tif",
    "flood_rp100_maxdepth_1km": OUT_R / f"{AOI}_flood_rp100_maxdepth_1km.tif",
}

rows = []
for k, p in derived_candidates.items():
    if p.exists():
        with rio.open(p) as ds:
            resx, resy = ds.res
            rows.append({
                "key": k,
                "path": str(p),
                "exists": True,
                "crs": str(ds.crs),
                "width": ds.width, "height": ds.height,
                "res_x": resx, "res_y": resy,
                "dtype": ds.dtypes[0],
                "nodata": ds.nodata
            })
    else:
        rows.append({"key": k, "path": str(p), "exists": False})

df_derived = pd.DataFrame(rows).sort_values("key")
df_derived
```

**This cell lists the Admin2 RAPP themes configured and their variable names.**

```{code-cell} ipython3
themes = [{"theme": t, "variables": ", ".join(THEME_VARS.get(t, []))} for t in ADMIN2_THEMES]
pd.DataFrame(themes).sort_values("theme")
```

**This cell checks if the expected Admin2 shapefile exists for each theme (per AOI naming).**

```{code-cell} ipython3
# Naming convention per your standard: data/vectors/ago_gov_{aoi}_{theme}_rapp_2020_a.shp
BASE = ROOT / "data" / "vectors"
rows = []
for t in ADMIN2_THEMES:
    shp = BASE / f"ago_gov_{AOI}_{t}_rapp_2020_a.shp"
    rows.append({"theme": t, "expected_path": str(shp), "exists": shp.exists()})
pd.DataFrame(rows).sort_values("theme")
```

**(Optional) This cell exports the two tables above as a tiny “data menu” CSV bundle for archiving.**

```{code-cell} ipython3
OUT_T = ROOT / "outputs" / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)
p1 = OUT_T / f"{AOI}_data_menu_core.csv"
p2 = OUT_T / f"{AOI}_data_menu_derived.csv"
df_core.to_csv(p1, index=False)
df_derived.to_csv(p2, index=False)
str(p1), str(p2)
```

---

## How to read the results (interpretation)

* **Core table**: if any `exists=False`, fix inputs or AOI naming in `config.py` before running steps.
* **Derived table**: **CRS** and **res_x/res_y** should match across 1-km rasters; mismatches imply stale/out-of-grid outputs.
* **RAPP themes**: confirm the themes you plan to use in Chapters 2, 4, and 7 actually exist for this AOI.

## Caveats

* Resolution units in EPSG:4326 are **degrees**; your 1-km grid approximates ~0.00833° at the equator (varies by latitude).
* Admin2 RAPP shapefiles must follow the **exact naming convention**; otherwise the quick check will show `exists=False`.

### Download

* (Optional) Data menu exports →
  `outputs/tables/{AOI}_data_menu_core.csv` and `outputs/tables/{AOI}_data_menu_derived.csv`

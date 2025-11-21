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
# 9. Change the knobs, not the code (practical tuning cheatsheet)

## Problem

We need to **tune** the analysis—without touching the scripts—to test whether our priorities are robust when data are thin or proxies (NTL/VEG) are noisy.

## Strategy

Expose the small set of **human-readable parameters** in `config.py` that control signal mix, masks, selection envelope, and clustering. Change them once, **re-run Steps 07 → 10 (and 11 if clusters are needed)**, and let all chapters refresh from `/outputs`.

## Data

This chapter reads no new data. It only **inspects**:

* Current parameters from `config.py` (read-only display)
* Scenario summary (if you ran Step 10)

## Methods (brief)

* **Weights**: control how much each layer contributes to the priority score (e.g., `W_POP`, `W_NTL`, `W_VEG`, `W_DRT`).
* **Masks & thresholds**: filter or emphasize target contexts (e.g., `MASK_MIN_CROPLAND`, rural/urban, electrification).
* **Selection envelope**: choose **Top-%** of cells or a **fixed km²** envelope.
* **Coherence**: remove speckles by raising `MIN_CLUSTER_CELLS`/`MIN_CLUSTER_KM2`; optional smoothing via `GAUSS_SIGMA_CELLS`.
* **Scenarios**: Step 10 bundles parameter sets (via a local `SCENARIOS` list) and summarizes **stability/swing**.

## Outputs

This page **doesn’t write** outputs. It only helps you preview what’s set and what scenarios were run.

---

## The knobs (what to change and why)

| What you want to adjust        | Parameter key(s) (in `config.py`)                 | Why you’d change it                                        |
| ------------------------------ | ------------------------------------------------- | ---------------------------------------------------------- |
| Make people dominate the score | `W_POP` ↑                                         | When poverty/need is the main driver and proxies are noisy |
| De-emphasize proxy signals     | `W_NTL` ↓, `W_VEG` ↓                              | When NTL/VEG bias toward better-served areas               |
| Penalize drought exposure      | `W_DRT` ↑ (or ↓)                                  | Stress-test climate sensitivity                            |
| Focus on production areas      | `MASK_MIN_CROPLAND` ↑                             | Avoid scattered rural cells with little cropland           |
| Focus on rural cells           | `MASK_REQUIRE_RURAL=True`                         | Keep smallholder / non-urban focus                         |
| Make hotspots larger/coherent  | `MIN_CLUSTER_CELLS` ↑                             | Remove speckles; ease deployment                           |
| Smooth noisy rasters           | `SMOOTH_RADIUS` ↑ *(0→no, 1→3×3, 2→5×5)*          | Reduce salt-and-pepper, especially with mixed sources      |
| Fix the total area of focus    | `TOP_KM2=…` (and set `TOP_PCT_CELLS=None`)        | Budget-like envelope for comparison across AOIs            |
| Use a percentage envelope      | `TOP_PCT_CELLS=…` (and set `TOP_KM2=None`)        | Relative selection when AOIs differ in size                |
| Adjust equity overlays         | `W_POV`, `W_FOOD`, `W_MTT`, `W_RWI`               | Tilt toward poverty, food insecurity, remoteness, or RWI   |

> **Where to edit?** Open `src/config.py`, find the `PARAMS` block, and change the values. No script edits needed.

---

## How to run (analyst)

After changing parameters in `config.py`, **re-run**:

* **Step 07** (priority surface + Admin2 tables) → **Step 10** (scenario summary, if used) → **Step 11** (clusters, if you need updated clusters).
  Chapters will automatically **load** the refreshed tables/rasters from `/outputs`.

**This cell prints a compact view of the active parameters (read-only).**

```{code-cell} ipython3
import os
from pathlib import Path
import pprint
from dataclasses import asdict

# Go up two levels (../..) to get from /docs/chapters/ to the repo root
ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")

# We only read/print; parameters are defined in src/config.py
import sys
sys.path.append(str(ROOT / "src"))
from config import PARAMS

print("AOI:", AOI)
print("Active parameters (subset):")
pp = pprint.PrettyPrinter(width=100, compact=True)

params = asdict(PARAMS)
show_keys = [
    "W_ACC","W_POP","W_VEG","W_NTL","W_DRT",
    "MASK_REQUIRE_RURAL","MASK_MIN_CROPLAND",
    "SMOOTH_RADIUS",
    "TOP_PCT_CELLS","TOP_KM2",
    "MIN_CLUSTER_CELLS",
    "SYNERGY_RADII_KM",
    "W_POV","W_FOOD","W_MTT","W_RWI",
]
filtered = {k: params.get(k) for k in show_keys}
pp.pprint(filtered)
```

**This cell lists any scenarios you’ve already run (from Step 10 summary).**

```{code-cell} ipython3
import pandas as pd

OUT_T = ROOT / "outputs" / "tables"
summary_path = OUT_T / f"{AOI}_priority_scenarios_summary.csv"

if summary_path.exists():
    summary = pd.read_csv(summary_path)
    print("Scenarios:", sorted(summary["scenario_id"].unique().tolist()))
else:
    print("No scenario summary found; run Step 10 to generate.")
```

**This cell reminds you which steps to re-run after changing parameters (text-only helper).**

```{code-cell} ipython3
print(
    "After editing PARAMS in src/config.py:\n"
    "  1) Run step_07_priority_tunable.py\n"
    "  2) (Optional) Run step_10_priority_scenarios.py\n"
    "  3) Run step_11_priority_clusters.py if you need updated clusters\n"
    "Open Chapters 2–4 again; they load from /outputs and will reflect the new results."
)
```

---

## How to read the results (interpretation)

* If **Top-K municipalities** barely change across scenarios, your picks are **robust**.
* If clusters become tiny/speckled, raise **coherence** knobs (`MIN_CLUSTER_*`) or use **Top-km²**.
* If equity correlation is weak, **de-emphasize proxies** (`W_NTL`, `W_VEG`) or raise **cropland threshold**.
* Always pair a retune with a one-liner rationale (e.g., “dropped NTL to avoid urban bias”).

## Caveats

* Changing multiple knobs at once can obscure cause/effect—prefer **small, named scenarios**.
* Keep a copy of the **parameter block** used in headline slides (Provenance section).

### Download

No downloads here—this is a **cheatsheet**. See Chapters 2–4 for the result tables those knobs influence.

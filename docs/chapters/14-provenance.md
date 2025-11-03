# 14. Provenance & reproducibility

## Problem

Decision-makers (and future you) need to know **exactly what was run**—which **AOI**, which **parameters**, and which **files** produced the results on this date.

## Strategy

Create a compact, machine-readable **provenance stamp** that includes: AOI, environment, parameter subset, scenario names, and **hashes/mtimes** for key outputs. Save it alongside outputs so any slide or memo can cite it.

## Data

No new data—this page only **reads** `config.py` and the existing files in `outputs/`.

## Methods (brief)

* Read `AOI`, `PARAMS` (subset), scenario list (if any).
* For key outputs, record **path, size, modified time, md5**.
* Save a single JSON file with all of the above.

## Outputs

* `outputs/tables/{AOI}_provenance.json` (created by the last cell below)

---

## How to run (analyst)

**This cell loads AOI, paths, and parameter subset from `config.py` (read-only).**

```python
import os, sys, platform, getpass, socket
from pathlib import Path
import pprint as _pp

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
sys.path.append(str(ROOT / "src"))

from config import PARAMS

print("AOI:", AOI)
print("ROOT:", ROOT)
pp = _pp.PrettyPrinter(width=100, compact=True)
print("Parameter subset (priority & selection relevant):")
keep = ["W_POP","W_NTL","W_VEG","W_DRT",
        "MASK_MIN_CROPLAND","MASK_URBAN_EXCLUDE","MASK_ELEC_EXISTING",
        "TOP_PCT_CELLS","TOP_KM2","MIN_CLUSTER_CELLS","MIN_CLUSTER_KM2",
        "GAUSS_SIGMA_CELLS"]
pp.pprint({k: PARAMS.get(k, None) for k in keep})
```

**This cell checks which scenario summary is present and lists scenario names (if any).**

```python
import pandas as pd

OUT_T = ROOT / "outputs" / "tables"
sum_path = OUT_T / f"{AOI}_priority_scenarios_summary.csv"
if sum_path.exists():
    summary = pd.read_csv(sum_path)
    scenarios = sorted(summary["scenario"].unique().tolist())
    print("Scenarios found:", scenarios)
else:
    scenarios = None
    print("No scenario summary found.")
```

**This cell defines the list of “headline files” to stamp (edit if you need more/less).**

```python
OUT_R = ROOT / "outputs" / "rasters"
headline = [
    OUT_T / f"{AOI}_priority_muni_rank.csv",
    OUT_T / f"{AOI}_priority_clusters.csv",
    OUT_T / f"{AOI}_catchments_kpis.csv",
    OUT_T / f"{AOI}_synergy_sites.csv",
    OUT_T / f"{AOI}_synergy_clusters.csv",
    OUT_T / f"{AOI}_priority_scenarios_summary.csv",   # may not exist
    OUT_T / f"{AOI}_od_flows.csv",                      # optional
    OUT_R / f"{AOI}_priority_clusters_1km.tif",
    OUT_R / f"{AOI}_priority_top10_mask.tif",           # or your Top-km² name
]
[p.name for p in headline]
```

**This cell computes size/mtime/md5 for each headline file that exists.**

```python
import hashlib, time

def _md5(path, block=2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def _stamp(path: Path):
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime)),
        "md5": _md5(path),
    }

stamps = [_stamp(p) for p in headline]
stamps
```

**This cell assembles the full provenance record and prints a compact preview.**

```python
from datetime import datetime, timezone

prov = {
    "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "host": {"node": socket.gethostname(), "platform": platform.platform()},
    "user": getpass.getuser(),
    "project_root": str(ROOT),
    "aoi": AOI,
    "parameters": {k: PARAMS.get(k, None) for k in keep},
    "scenarios": scenarios,
    "artifacts": stamps,
}

pp = _pp.PrettyPrinter(width=100, compact=True)
pp.pprint({
    "generated_utc": prov["generated_utc"],
    "aoi": prov["aoi"],
    "parameters": prov["parameters"],
    "scenarios": prov["scenarios"],
    "n_artifacts": len(prov["artifacts"]),
})
```

**This cell saves the provenance record to `/outputs/tables/{AOI}_provenance.json`.**

```python
import json

OUT_T.mkdir(parents=True, exist_ok=True)
prov_path = OUT_T / f"{AOI}_provenance.json"
with open(prov_path, "w", encoding="utf-8") as f:
    json.dump(prov, f, ensure_ascii=False, indent=2)
prov_path
```

---

## How to read the results (interpretation)

* **generated_utc** is the time you stamped the result set; cite it in slides.
* **parameters** is the exact mix that produced the current outputs—paste into annexes.
* **artifacts** list lets you match files by **md5** if there are multiple versions floating around.
* **scenarios** confirm which named scenarios were included in Chapter 4 comparisons.

## Caveats

* If you **rerun** steps after stamping, regenerate the provenance to keep hashes in sync.
* Host/user info is included for transparency; remove those fields if you need a neutral record.
* If any headline file is **missing**, decide whether to exclude it from the stamp or to rerun the step.

### Download

* Provenance JSON → `outputs/tables/{AOI}_provenance.json`

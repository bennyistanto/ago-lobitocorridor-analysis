# 7. Do the priorities align with poverty & food insecurity? (equity lens)

## Problem

Are we prioritizing municipalities and hotspots that also have **higher rural poverty** and **greater food insecurity**—or are we inadvertently biasing toward better-off places?

## Strategy

Join the **priority results** to **RAPP** socio-economic themes (poverty, food insecurity) at **Admin2** level, then check simple equity diagnostics: **correlations**, **rank overlays**, and **outlier flags** (high priority + low poverty, or vice versa).

## Data

* **Municipality ranking** — `outputs/tables/{AOI}_priority_muni_rank.csv` (Step 09; Step 06 attributes merged in)
* **RAPP themes (Admin2)** — Step 06 ingest (poverty, food insecurity) merged into the rank table
* *(Optional)* Priority clusters — `outputs/tables/{AOI}_priority_clusters.csv` (for cluster-level anecdotes)

## Methods (brief)

* Merge **priority score** per municipality with **rural poverty** and **food-insecurity** indicators (from RAPP).
* Compute **Pearson r** (score vs rural poverty; score vs food insecurity).
* List **equity outliers**: municipalities with top priority but bottom-half poverty (and the reverse).
* (Optional) Note which **priority clusters** fall inside high-poverty municipalities.

## Outputs

* Equity diagnostics are **displayed in this chapter** from the already-saved tables; no new files required.
* *(Optional)* Save a light extract for slides as `outputs/tables/{AOI}_equity_lens_extract.csv` (see last cell).

## How to run (analyst)

No recomputation here. Ensure Steps **06** and **09** have been run so `/outputs/tables/{AOI}_priority_muni_rank.csv` contains the merged RAPP attributes.

**This cell loads the municipality ranking (with RAPP attributes) from `/outputs`.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT  = ROOT / "outputs" / "tables"

rank = pd.read_csv(OUT / f"{AOI}_priority_muni_rank.csv")
rank.head(10)
```

## Quick results

**This cell lists the columns available so you can verify poverty/food fields are present.**

```{code-cell} ipython3
rank.columns.tolist()
```

**This cell runs the equity correlations (score vs rural poverty; score vs food insecurity) with auto column detection.**

```{code-cell} ipython3
# Robust column detection (adjust candidates if your merge used different names)
pov_candidates  = ["rural_poverty", "poverty_rural", "RURAL_POV", "data1"]  # 'data1' only if mapped that way
food_candidates = ["food_insec_scale", "food_insecurity", "FOOD_INSEC", "data9"]  # 'data9' if mapped

pov_col  = next((c for c in pov_candidates  if c in rank.columns), None)
food_col = next((c for c in food_candidates if c in rank.columns), None)

results = {}
if pov_col:
    results["r_score_poverty"] = round(rank["score"].corr(rank[pov_col]), 3)
if food_col:
    results["r_score_food"]    = round(rank["score"].corr(rank[food_col]), 3)

results if results else "No poverty/food columns found; revisit Step 06 join."
```

**This cell flags equity outliers: (A) high priority but low poverty; (B) high poverty but low priority.**

```{code-cell} ipython3
# Define halves or quantiles as your policy prefers
q_score_hi = rank["score"].quantile(0.75)  # top quartile by priority
q_score_lo = rank["score"].quantile(0.25)  # bottom quartile
out = {}

if pov_col:
    q_pov_hi = rank[pov_col].quantile(0.75)   # high poverty
    q_pov_lo = rank[pov_col].quantile(0.25)   # low poverty

    out["A_high_priority_low_poverty"] = (
        rank.loc[(rank["score"] >= q_score_hi) & (rank[pov_col] <= q_pov_lo),
                 ["NAM_1","NAM_2","score",pov_col]]
        .sort_values("score", ascending=False)
        .head(15)
    )

    out["B_high_poverty_low_priority"] = (
        rank.loc[(rank[pov_col] >= q_pov_hi) & (rank["score"] <= q_score_lo),
                 ["NAM_1","NAM_2","score",pov_col]]
        .sort_values(pov_col, ascending=False)
        .head(15)
    )

out if out else "Cannot compute outliers (poverty column missing)."
```

**(Optional) This cell shows a small scatter of score vs. poverty for a quick visual check.**

```{code-cell} ipython3
import matplotlib.pyplot as plt

if pov_col:
    plt.figure()
    plt.scatter(rank[pov_col], rank["score"], s=12)
    plt.xlabel("Rural poverty rate (%)")
    plt.ylabel("Composite priority score (0–1)")
    plt.title(f"{AOI}: Do priorities align with poverty?")
    plt.show()
else:
    print("Poverty column not found; skipping scatter.")
```

**(Optional) This cell saves a compact extract for slides (top-10 by score with poverty & food fields).**

```{code-cell} ipython3
keep_cols = ["NAM_1","NAM_2","score"]
if pov_col:  keep_cols.append(pov_col)
if food_col: keep_cols.append(food_col)

extract = rank.sort_values("score", ascending=False).head(10)[keep_cols]
extract_path = OUT / f"{AOI}_equity_lens_extract.csv"
extract.to_csv(extract_path, index=False)
extract_path
```

## How to read the results (interpretation)

* **Positive correlation** (score vs. rural poverty) suggests priorities are skewing toward **poorer municipalities**—often desirable.
* **Weak/negative correlation** is not automatically bad; it may mean proxies (e.g., NTL/VEG) favor less-poor places—use Chapter 4 scenarios to retune.
* **Outliers A (high priority, low poverty)**: plausible logistics wins—flag for justification or rebalance.
* **Outliers B (high poverty, low priority)**: consider whether access or production constraints hide need—explore targeted fixes.

## Caveats

* RAPP poverty and food-insecurity indicators are **model/survey-based**; small-area noise happens.
* Priority reflects the **current weight mix**; check Chapter 4’s stability before messaging.
* Equity checks at Admin2 can **mask intra-municipality pockets**; combine with cluster-level views (Chapter 3).

### Download

* Municipality **ranking** → `outputs/tables/{AOI}_priority_muni_rank.csv`
* (Optional) Equity extract for slides → `outputs/tables/{AOI}_equity_lens_extract.csv`

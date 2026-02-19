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
* Compute a **benefit incidence curve** (cumulative benefit vs. cumulative population, ranked poorest→richest) and a **concentration index** (CI) that summarizes pro-poor or pro-rich targeting in a single number.
* List **equity outliers**: municipalities with top priority but bottom-half poverty (and the reverse).
* (Optional) Note which **priority clusters** fall inside high-poverty municipalities.

## Outputs

* `outputs/tables/{AOI}_benefit_incidence.csv` — benefit incidence curve data (ADM2, poverty, cumulative shares)
* `outputs/tables/{AOI}_equity_summary.csv` — concentration index, interpretation, totals
* Equity diagnostics are **displayed in this chapter** from the already-saved tables; no new files required.
* *(Optional)* Save a light extract for slides as `outputs/tables/{AOI}_equity_lens_extract.csv` (see last cell).

## How to run (analyst)

No recomputation here. Ensure Steps **06** and **09** have been run so `/outputs/tables/{AOI}_priority_muni_rank.csv` contains the merged RAPP attributes.

**This cell loads the municipality ranking (with RAPP attributes) from `/outputs`.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

# Go up two levels (../..) to get from /docs/chapters/ to the repo root
ROOT = Path(os.getenv("PROJECT_ROOT", "../.."))
AOI  = os.getenv("AOI", "huambo")
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

**This cell loads the concentration index — a single number that summarizes whether benefits are pro-poor.**

```{code-cell} ipython3
eq_path = OUT / f"{AOI}_equity_summary.csv"
bi_path = OUT / f"{AOI}_benefit_incidence.csv"

if eq_path.exists():
    eq = pd.read_csv(eq_path)
    ci = eq["concentration_index"].iloc[0]
    interp = eq["interpretation"].iloc[0]
    print(f"Concentration Index (CI) = {ci:.4f}")
    print(f"Interpretation: {interp}")
    print()
    print("What this means:")
    print("  CI > 0  → Benefits flow to poorer municipalities (pro-poor)")
    print("  CI < 0  → Benefits flow to richer municipalities (pro-rich)")
    print("  CI ≈ 0  → Benefits distributed roughly evenly across income levels")
else:
    print("Equity summary not found; run Step 09.")
```

**This cell plots the benefit incidence curve — a visual equity diagnostic for presentations.**

```{code-cell} ipython3
if bi_path.exists():
    bi = pd.read_csv(bi_path)
    plt.figure(figsize=(7, 5))
    plt.fill_between(bi["cum_pop_share"], bi["cum_benefit_share"], bi["cum_pop_share"],
                     where=(bi["cum_benefit_share"] >= bi["cum_pop_share"]),
                     alpha=0.2, color="green", label="Pro-poor zone")
    plt.fill_between(bi["cum_pop_share"], bi["cum_benefit_share"], bi["cum_pop_share"],
                     where=(bi["cum_benefit_share"] < bi["cum_pop_share"]),
                     alpha=0.2, color="red", label="Pro-rich zone")
    plt.plot(bi["cum_pop_share"], bi["cum_benefit_share"], "b-o", ms=4, label="Benefit incidence")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect equality")
    plt.xlabel("Cumulative population share (poorest → richest)")
    plt.ylabel("Cumulative benefit share")
    plt.title(f"{AOI}: Are investment benefits reaching the poor?")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("Benefit incidence table not found; run Step 09.")
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
* **Concentration index > 0** is the strongest evidence of pro-poor targeting. Unlike correlation, the CI accounts for population sizes and benefit magnitudes, not just ranks.
* **Benefit incidence curve above the diagonal** means poorer municipalities receive a larger share of benefits than their share of population — the ideal outcome.
* **Weak/negative correlation** is not automatically bad; it may mean proxies (e.g., NTL/VEG) favor less-poor places—use Chapter 4 scenarios to retune.
* **Outliers A (high priority, low poverty)**: plausible logistics wins—flag for justification or rebalance.
* **Outliers B (high poverty, low priority)**: consider whether access or production constraints hide need—explore targeted fixes.

## Caveats

* RAPP poverty and food-insecurity indicators are **model/survey-based**; small-area noise happens.
* Priority reflects the **current weight mix**; check Chapter 4’s stability before messaging.
* Equity checks at Admin2 can **mask intra-municipality pockets**; combine with cluster-level views (Chapter 3).

### Download

* Municipality **ranking** → `outputs/tables/{AOI}_priority_muni_rank.csv`
* Benefit **incidence curve** → `outputs/tables/{AOI}_benefit_incidence.csv`
* **Equity summary** (concentration index) → `outputs/tables/{AOI}_equity_summary.csv`
* (Optional) Equity extract for slides → `outputs/tables/{AOI}_equity_lens_extract.csv`

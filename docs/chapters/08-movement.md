# 8. Main movement patterns

## Problem

Where do people and goods **want to move** across municipalities in the Lobito Corridor under current conditions—and which **origin–destination pairs** would benefit most from last-mile fixes?

## Strategy

Use a light-weight **gravity model** at Admin2 level (population × opportunities × distance-decay), optionally filtered by **market access** and **cropland**, then (a) summarize **OD flows** and (b) sample **agent trips** for desire-lines.

## Data

* **Admin2 centroids / zone grid** — from Step 06 ingest
* **Population (WorldPop, 1-km)** — Step 00, aggregated to Admin2
* **Access / opportunities** — e.g., inverse travel time to market, or market score (Step 00/07)
* **Road / rail context** *(map only; not needed to load here)*

## Methods (brief)

* Build an **OD matrix** between zones using a **gravity model** with tunable parameters *(α, β, λ, γ)* on mass, impedance, and balancing.
* Normalize flows to a target total (e.g., 1.0) or daily trips.
* **Sample agents** proportionally to OD weights (optional), producing a CSV of origin/destination coordinates for visual desire-lines.
  *(All computed in Step 14; this chapter only loads and inspects outputs.)*

## Outputs

* `outputs/tables/{AOI}_od_gravity.csv`  
  *(columns: `oi`, `dj`, `flow`, `dist_km`; where `oi`/`dj` are Admin2 codes, usually `ADM2CD_c`)*
* `outputs/tables/{AOI}_od_zone_attrs.csv`  
  *(columns: `ADM2CD_c`, `NAM_1`, `NAM_2`, `lon`, `lat`, `pop`, optional: `rwi_z`, `mass`)*  
* `outputs/tables/{AOI}_od_agents.csv` *(optional)*  
  *(columns: `agent_id`, `o_lon`, `o_lat`, `d_lon`, `d_lat`, optional: `weight`)*

Gravity flows and zone attributes are always written by **Step 14**; agent sampling can be disabled by setting `OD_N_AGENTS=0` in `config.PARAMS` if needed.

## How to run (analyst)

Run **Step 14** once to create the OD tables. This chapter only **loads** saved outputs (no recomputation).

**This cell loads OD flows and (if present) sampled agents for the current AOI.**

```{code-cell} ipython3
import os
import pandas as pd
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", "."))
AOI  = os.getenv("AOI", "moxico")
OUT_T = ROOT / "outputs" / "tables"

flows_path  = OUT_T / f"{AOI}_od_gravity.csv"
zones_path  = OUT_T / f"{AOI}_od_zone_attrs.csv"
agents_path = OUT_T / f"{AOI}_od_agents.csv"

flows = pd.read_csv(flows_path)  if flows_path.exists()  else None
zones = pd.read_csv(zones_path)  if zones_path.exists()  else None
agents = pd.read_csv(agents_path) if agents_path.exists() else None

print("Loaded:",
      flows_path.name if flows is not None else "no od_gravity",
      "|",
      zones_path.name if zones is not None else "no od_zone_attrs",
      "|",
      agents_path.name if agents is not None else "no od_agents")
```

## Quick results

**This cell previews the top 15 OD pairs by modelled flow (biggest interaction).**

```{code-cell} ipython3
if (flows is not None) and (zones is not None):
    # Map Admin2 code → name
    z = zones.set_index("ADM2CD_c")
    flows_named = (
        flows.copy()
        .assign(
            o_adm2 = flows["oi"],
            d_adm2 = flows["dj"],
        )
        .merge(z[["NAM_2"]].rename(columns={"NAM_2": "o_name"}),
               left_on="o_adm2", right_index=True, how="left")
        .merge(z[["NAM_2"]].rename(columns={"NAM_2": "d_name"}),
               left_on="d_adm2", right_index=True, how="left")
    )

    top_pairs = flows_named.sort_values("flow", ascending=False).head(15)
    top_pairs[["o_adm2","o_name","d_adm2","d_name","flow","dist_km"]]
else:
    print("OD flows or zone attributes not found; run Step 14.")
```

**This cell aggregates flows by origin and by destination to see main senders/receivers.**

```{code-cell} ipython3
if (flows is not None) and (zones is not None):
    z = zones.set_index("ADM2CD_c")

    by_origin = (
        flows.groupby("oi", as_index=False)["flow"].sum()
             .rename(columns={"oi": "adm2", "flow": "outflow"})
             .merge(z[["NAM_2"]], left_on="adm2", right_index=True, how="left")
    ).sort_values("outflow", ascending=False)

    by_dest = (
        flows.groupby("dj", as_index=False)["flow"].sum()
             .rename(columns={"dj": "adm2", "flow": "inflow"})
             .merge(z[["NAM_2"]], left_on="adm2", right_index=True, how="left")
    ).sort_values("inflow", ascending=False)

    by_origin.head(10), by_dest.head(10)
else:
    print("OD flows or zone attributes not found; run Step 14.")
```

**This cell computes a simple centrality proxy: (inflow + outflow) per municipality.**

```{code-cell} ipython3
if (flows is not None) and (zones is not None):
    z = zones.set_index("ADM2CD_c")

    by_origin = flows.groupby("oi", as_index=False)["flow"].sum().rename(columns={"oi":"adm2","flow":"outflow"})
    by_dest   = flows.groupby("dj", as_index=False)["flow"].sum().rename(columns={"dj":"adm2","flow":"inflow"})

    cent = (
        by_origin.merge(by_dest, on="adm2", how="outer")
                 .fillna(0.0)
                 .assign(throughput=lambda d: d["inflow"] + d["outflow"])
                 .merge(z[["NAM_2"]], left_on="adm2", right_index=True, how="left")
                 .sort_values("throughput", ascending=False)
    )
    cent.head(15)
else:
    print("OD flows or zone attributes not found; run Step 14.")
```

**This cell shows a small sample of agent trips (if you saved them).**

```{code-cell} ipython3
if agents is not None:
    agents.head(10)
else:
    print("Agents sample not found; Step 14 may have skipped sampling.")
```

**This cell draws a quick bar of top-10 OD pairs by flow (labels truncated).**

```{code-cell} ipython3
import matplotlib.pyplot as plt

if (flows is not None) and (zones is not None):
    z = zones.set_index("ADM2CD_c")
    t10 = flows.nlargest(10, "flow").copy()
    t10 = (
        t10
        .merge(z[["NAM_2"]].rename(columns={"NAM_2":"o_name"}), left_on="oi", right_index=True, how="left")
        .merge(z[["NAM_2"]].rename(columns={"NAM_2":"d_name"}), left_on="dj", right_index=True, how="left")
    )
    t10["label"] = t10["o_name"].fillna(t10["oi"].astype(str)).str[:10] + " → " + \
                   t10["d_name"].fillna(t10["dj"].astype(str)).str[:10]

    plt.figure()
    plt.barh(t10["label"], t10["flow"])
    plt.gca().invert_yaxis()
    plt.xlabel("Modelled flow (relative units)")
    plt.title(f"{AOI}: Top OD pairs (gravity model)")
    plt.show()
else:
    print("OD flows or zone attributes not found; run Step 14.")
```

## How to read the results (interpretation)

* **Top OD pairs** highlight **corridors of interaction**: if they cross weak links (poor road class, known bottlenecks), they’re prime candidates for last-mile upgrades.
* **High-throughput municipalities** (inflow + outflow) act as **hubs**; improving access here can lift multiple OD pairs.
* **Distance effect:** very short high flows may already be well served; medium-distance high flows often reveal **latent demand** suppressed by access.

## Caveats

* Gravity flows are **modelled**—they signal interaction potential, not measured trips.
* Centroids/zone design matter; dense municipalities can dominate if not normalized.
* If access/opportunity inputs change (e.g., new market times), OD results should be **re-run**.

### Download

* OD **flows** → `outputs/tables/{AOI}_od_gravity.csv`
* OD **zone attributes** → `outputs/tables/{AOI}_od_zone_attrs.csv`
* OD **agents sample** (optional) → `outputs/tables/{AOI}_od_agents.csv`

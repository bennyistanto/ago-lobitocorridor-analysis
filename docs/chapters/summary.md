# Corridor-wide results & summary

This page summarizes the **main quantitative results** from the corridor
analysis. It is designed for decision makers and analysts who want to see
the *numbers, charts and maps* before diving into the detailed views in
Chapters 1–8.

The narrative here is **province-agnostic**: the same indicators are
reported for each AOI where the pipeline has been run (e.g. Huambo now,
additional provinces later). Differences across provinces come entirely
from the data, not from ad hoc methods.

> The flowchart below summarises the data sources, processing steps, and outputs used to generate the tables and figures in this section. It is mainly intended for readers who want to understand the underlying workflow.

```{mermaid}
---
config:
  layout: elk
---
flowchart TB
 subgraph Inputs["Inputs"]
        A1["Access data: roads & travel times"]
        A2["Poverty and food insecurity data"]
        A3["Population and cropland grids"]
        A4["Projects and candidate sites"]
  end
 subgraph Core_Analysis["Core analysis steps"]
        B1["Step 07: 1 km priority surface"]
        B2["Steps 06 & 09: Admin2 indicators and scores"]
        B3["Step 11: Priority clusters"]
        B4["Step 12: Catchments (30 / 60 / 120 min)"]
        B5["Step 13: Project synergies"]
        B6["Step 14: OD-lite flows & bottlenecks"]
  end
 subgraph Extended["Extended analysis"]
        D1["Step 09: Benefit incidence & equity index"]
        D2["Step 16: Intervention simulator"]
        D3["Step 15: Corridor dashboard"]
  end
 subgraph Outputs["Outputs"]
        C1["Summary tables S1–S9"]
        C2["Maps and dashboard views"]
  end
    A1 --> B1
    A2 --> B2
    A3 --> B1 & B4
    A4 --> B4 & B5
    B1 --> B2 & B3
    B2 --> B3 & B6 & D1
    B3 --> B4 & B5 & C2
    B4 --> C1 & C2
    B5 --> C1 & C2
    B6 --> C1 & C2 & D2
    D1 --> C1
    D2 --> C1 & C2
    D3 --> C1
```

---

## 1. Where do the strongest priority clusters emerge?

Across all provinces processed so far, the 1-km priority surface (Step 07)
reveals a small number of **high-intensity clusters** that concentrate
most of the potential beneficiaries. These clusters represent places where:

- multiple constraints stack (low baseline access, poor electrification,
  high rural poverty and/or food insecurity), and
- there is enough **population and cropland** to justify coordinated
  investments.

> Flowchart below provides a schematic of how the 1-km priority surface, population, and cropland data are combined to identify priority clusters and produce Table S1 and Figure S1.

```{mermaid}
flowchart TB
 subgraph S4_inputs["Inputs"]
        s4i1["1 km priority surface (Step 07)"]
        s4i2["Population grid"]
        s4i3["Cropland grid"]
        s4i4["Admin1/Admin2 boundaries"]
  end
 subgraph S4_process["Processing"]
        s4p1["Apply top 10% threshold to priority surface"]
        s4p2["Identify contiguous priority clusters"]
        s4p3["Compute cluster stats: area, people, cropland"]
        s4p4["Aggregate to province level for summary table"]
  end
 subgraph S4_outputs["Outputs for Summary 1"]
        s4o1["Table S1: priority clusters by province"]
        s4o2["Figure S1: map of priority clusters along corridor"]
        s4o3["Messages on where needs and opportunities stack"]
  end
    s4i1 --> s4p1
    s4i2 --> s4p3
    s4i3 --> s4p3
    s4i4 --> s4p2
    s4p1 --> s4p2
    s4p2 --> s4p3 & s4o2
    s4p3 --> s4p4
    s4p4 --> s4o1
    s4o1 --> s4o3
    s4o2 --> s4o3
```

**Table S1** below summarizes, for each province and its top clusters:

- the **total priority cluster area** in km²,
- the share of the province’s population located inside the priority clusters (the Top 10% surface after pruning small speckles),
- the share of cropland inside the priority cluster, and
- the number of distinct clusters.

> **Table S1. Priority clusters by province (illustrative structure)**  
> *Generated from `{AOI}_priority_clusters.csv` and
> `{AOI}_kpis_isochrones.csv`.*

| Province (AOI) | Number of priority clusters | Priority cluster area (km²) | Population in clusters (people) | Total population (province, people)  | Population in clusters (% of province total) | Cropland in clusters (km²) | Total cropland (province, km²) | Cropland in clusters (% of province total cropland) | Population density in clusters (people/km²) | Cropland share of cluster area (km² cropland per km² area) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Benguela | 4 | 431.69 | 3134.0 | 3271805.25 | 0.1 | 217.2 | 5749.06 | 3.78 | 7.26 | 0.5 |
| Huambo | 2 | 52.86 | 31776.0 | 2997196.75 | 1.06 | 34.72 | 10156.91 | 0.34 | 601.14 | 0.66 |
| Bie      | 1  | 26.89 | 18307.0 | 2159939.0 | 0.85 | 11.17 | 5923.75 | 0.19 | 680.81 | 0.42 |
| Moxico | 0  | 0.0  | 0.0  | 756128.37  | 0.0 | 0.0 | 346.98 | 0.0 | NaN | NaN |
| Moxico Leste | 0 | 0.0  | 0.0  | 372044.31  | 0.0 | 0.0 | 51.86 | 0.0 | NaN | NaN |

**Figure S1** shows the **spatial pattern of priority clusters** along the
corridor: clusters hug the rail and primary road spine in a few key
segments, rather than being evenly spread across all municipalities.

> **Figure S1. Corridor-wide priority clusters**  
> *Static map showing priority clusters (Step 11) overlaid on the
> Lobito Corridor rail line and key roads, with provincial boundaries.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S1-Benguela](../../outputs/figs/AGO_A4L_LC_Benguela_PriorityArea.png)
:::

:::{tab-item} Huambo
:sync: key2
![S1-Huambo](../../outputs/figs/AGO_A4L_LC_Huambo_PriorityArea.png)
:::

:::{tab-item} Bie
:sync: key3
![S1-Bie](../../outputs/figs/AGO_A4L_LC_Bie_PriorityArea.png)
:::

:::{tab-item} Moxico
:sync: key4
![S1-Moxico](../../outputs/figs/AGO_A4L_LC_Moxico_PriorityArea.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S1-MoxicoLeste](../../outputs/figs/AGO_A4L_LC_MoxicoLeste_PriorityArea.png)
:::
::::

---

## 2. Are we focusing where needs and opportunities coincide?

The Admin2 analysis deliberately balances **need** and **opportunity**, combining the tunable priority raster (Step 07) with municipality indicators from the RAPP survey (Step 06 / Step 09).

- **Need** is captured through rural poverty and, where available,
  food insecurity, long travel times and low electrification.
- **Opportunity** is captured through how much of each municipality lies
  inside the priority mask and how many people and hectares are exposed
  to improved access if we invest there.

For each province, we compute a **composite Admin2 score** that combines:

- the share of the municipality covered by the priority mask,
- rural poverty and food insecurity indices (RAPP-based, where available),
- average travel time to markets or services,
- electrification and, optionally, a wealth proxy (RWI).

This gives a single 0–1 score that can be compared across municipalities
within the same province.

> Flowchart below outlines how municipality-level indicators are normalised and combined into a composite score, and how this links to the quadrant analysis of priority score versus rural poverty used for Table S2 and Figure S2.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S5_inputs["Inputs"]
        s5i1["Admin2 indicators: poverty, food insecurity, travel time, electrification, RWI"]
        s5i2["Share of municipality in priority mask"]
        s5i3["Admin2 lookup and names"]
  end
 subgraph S5_process["Processing"]
        s5p1["Normalise indicators to common 0 to 1 scale"]
        s5p2["Compute composite Admin2 score"]
        s5p3["Rank municipalities within each province"]
        s5p4["Compare score vs rural poverty to define quadrants"]
        s5p5["Estimate rural poor by municipality"]
        s5p6["Summarise rural poor shares by quadrant"]
  end
 subgraph S5_outputs["Outputs for Summary 2"]
        s5o1["Table S2: top municipalities by composite score"]
        s5o2["Figure S2: score vs rural poverty scatter"]
        s5o3["Quadrant summary: counts and share of rural poor"]
        s5o4["Equity messages: alignment and under prioritised areas"]
  end
    s5i1 --> s5p1
    s5i2 --> s5p2
    s5i3 --> s5p3
    s5p1 --> s5p2
    s5p2 --> s5p3 & s5p4
    s5p4 --> s5p5 & s5o2
    s5p5 --> s5p6
    s5p3 --> s5o1
    s5p6 --> s5o3
    s5o1 --> s5o4
    s5o2 --> s5o4
    s5o3 --> s5o4
```

**Table S2** highlights, for each province, the **top-ranked municipalities**
and how they compare against the provincial average on key equity and
access indicators.

> **Table S2. Top 5 municipalities by composite score (illustrative structure)**  
> *Generated from `{AOI}_priority_muni_rank.csv` (one file per province) and
> aggregated into a single corridor-wide table.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| benguela | Catumbela  | 0.702 | 0.353 | 0.672 | 30.519  | 54.1 | 0.000  |
| benguela | Benguela   | 0.683 | 0.349 | 0.666 | 49.682  | 15.0 | 0.000  |
| benguela | Caimbambo  | 0.646 | 0.596 | 0.790 | 131.391 | 10.4 | 7.380  |
| benguela | Cubal      | 0.629 | 0.740 | 0.764 | 178.319 | 6.1  | 70.097 |
| benguela | Baia Farta | 0.629 | 0.442 | 0.673 | 75.299  | 50.6 | 0.000  |

:::

:::{tab-item} Huambo
:sync: key2

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Huambo | Ekunha     | 0.708 | 0.683 | 0.685 | 61.163  | 16.3  | 0.000   |
| Huambo | Huambo     | 0.703 | 0.509 | 0.675 | 39.249  | 12.0  | 50.827  |
| Huambo | Caala      | 0.665 | 0.635 | 0.663 | 71.650  | 37.6  | 49.173  |
| Huambo | Bailundo   | 0.619 | 0.608 | 0.626 | 105.837 | 26.7  | 0.000   |
| Huambo | Katchiungo | 0.608 | 0.649 | 0.630 | 91.717  | 51.9  | 0.000   |

:::

:::{tab-item} Bie
:sync: key3

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| bie | Camacupa | 0.706  | 0.594  | 0.692  | 245.867  | 1.3  | 0.0   |
| bie | Kuito    | 0.692  | 0.483  | 0.701  | 162.152  | 5.7  | 100.0 |
| bie | N'harea  | 0.689  | 0.495  | 0.609  | 41.423   | 4.6  | 0.0   |
| bie | Andulo   | 0.653  | 0.530  | 0.644  | 214.413  | 4.4  | 0.0   |
| bie | Catabola | 0.627  | 0.490  | 0.648  | 201.453  | 2.3  | 0.0   |

:::

:::{tab-item} Moxico
:sync: key4

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico | Luena (moxico)   | 0.748  | 0.674  | 0.640 | 145.567  | 4.9  | NaN |
| Moxico | Lumbala N'guimbo | 0.549  | 0.740  | 0.684 | 484.487  | 1.6  | NaN |
| Moxico | Luchazes         | 0.434  | 0.675  | 0.693 | 327.545  | 5.1  | NaN |
| Moxico | Camanongue       | NaN    | 0.728  | 0.738 | 101.129  | 4.5  | NaN |
| Moxico | Leua             | NaN    | 0.627  | 0.784 | 75.766   | 18.5 | NaN |

:::

:::{tab-item} Moxico Leste
:sync: key5

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico Leste | Alto Zambeze | 0.732 | 0.739 | 0.647 | 196.08 | 4.3 | NaN |

:::
::::

To test whether the composite score actually aligns with equity objectives,
we compare **priority score vs. rural poverty** across all provinces.

**Figure S2** shows, for every municipality where data are available:

- the composite priority score on the x-axis,
- the rural poverty index on the y-axis, and
- points colored by province.

The dashed lines mark the median score and median rural poverty across
the corridor, dividing municipalities into four quadrants:

- **high score & high poverty** (ideal alignment),
- **high score & lower poverty** (valid but efficiency-driven),
- **lower score & high poverty** (potentially “missed” poor municipalities),
- **lower score & lower poverty** (naturally deprioritized).

> **Figure S2. Priority score vs. rural poverty, all municipalities**  
> *Scatter plot built from all `{AOI}_priority_muni_rank.csv` files, with points
> colored by province and a small number of outliers labelled by Admin2 name.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S2-Benguela](../../outputs/figs/benguela_fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| Quadrant (score x poverty)  | Number of municipalities | Estimated rural poor (people) | Share of province’s rural poor (%) |
| ---- | ---- | ---- | ---- |
| High score & high poverty   | 1  | 96202.0  | 37.0 |
| High score & lower poverty  | 4  | 44197.0  | 17.0 |
| Lower score & high poverty  | 4  | 76396.0  | 30.0 |
| Lower score & lower poverty | 1  | 41861.0  | 16.0 |

- About 1 of 10 municipalities (10.0%) fall in the **high score & high poverty** quadrant, representing roughly 37% of the estimated rural poor in Benguela covered by the dataset.
- Around 4 municipalities (40.0%) sit in the **lower score & high poverty** quadrant, accounting for about 30% of the estimated rural poor — these are potentially under-prioritized areas.

:::

:::{tab-item} Huambo
:sync: key2
![S2-Huambo](../../outputs/figs/huambo_fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| Quadrant (score x poverty)  | Number of municipalities | Estimated rural poor (people) | Share of province’s rural poor (%) |
| ---- | ---- | ---- | ---- |
| High score & high poverty   | 3  | 57798.0  | 18.0 |
| High score & lower poverty  | 3  | 55377.0  | 18.0 |
| Lower score & high poverty  | 3  | 43169.0  | 14.0 |
| Lower score & lower poverty | 2  | 157305.0 | 50.0 |

- About 3 of 11 municipalities (27.3%) fall in the **high score & high poverty** quadrant, representing roughly 18.4% of the estimated rural poor within Huambo” vs “covered by the dataset.
- Around 3 municipalities (27.3%) sit in the **lower score & high poverty** quadrant, accounting for about 13.8% of the estimated rural poor — these are potentially under-prioritized areas.

:::

:::{tab-item} Bie
:sync: key3
![S2-Bie](../../outputs/figs/bie_fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| Quadrant (score x poverty)  | Number of municipalities | Estimated rural poor (people) | Share of province’s rural poor (%) |
| ---- | ---- | ---- | ---- |
| High score & high poverty   | 2  | 22410.0  | 20.0 |
| High score & lower poverty  | 2  | 42326.0  | 38.0 |
| Lower score & high poverty  | 2  | 19672.0  | 18.0 |
| Lower score & lower poverty | 2  | 27253.0  | 24.0 |

- About 2 of 8 municipalities (25.0%) fall in the **high score & high poverty** quadrant, representing roughly 20% of the estimated rural poor in Bie covered by the dataset.
- Around 2 municipalities (25.0%) sit in the **lower score & high poverty** quadrant, accounting for about 18% of the estimated rural poor — these are potentially under-prioritized areas.
:::

:::{tab-item} Moxico
:sync: key4
![S2-Moxico](../../outputs/figs/moxico_fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| Quadrant (score x poverty)  | Number of municipalities | Estimated rural poor (people) | Share of province’s rural poor (%) |
| ---- | ---- | ---- | ---- |
| High score & high poverty  | 1 | 333.0  | 10.0 |
| High score & lower poverty | 1 | 3116.0 | 90.0 |
| Lower score & high poverty | 1 | 15.0   | 0.0  |

- About 1 of 3 municipalities (33.3%) fall in the **high score & high poverty** quadrant, representing roughly 10% of the estimated rural poor in Moxico covered by the dataset.
- Around 1 municipalities (33.3%) sit in the **lower score & high poverty** quadrant, accounting for about 0% of the estimated rural poor — these are potentially under-prioritized areas.
:::

:::{tab-item} Moxico Leste
:sync: key5
![S2-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| Quadrant (score x poverty)  | Number of municipalities | Estimated rural poor (people) | Share of province’s rural poor (%) |
| ---- | ---- | ---- | ---- |
| High score & high poverty  | 1 | 620.0  | 100.0 |

- About 1 of 1 municipalities (100.0%) fall in the **high score & high poverty** quadrant, representing roughly 100% of the estimated rural poor in Moxicoleste covered by the dataset.
:::
::::

---

## 3. How many people and hectares benefit within 30/60/120 minutes?

For each candidate site (e.g. existing or potential investments) and
province, the catchment analysis (Step 12) quantifies **how many people
and how much cropland** are reachable within 30, 60, and 120 minutes
along the existing and improved network.

This allows us to compare:

- sites that reach **many people quickly** (high 30-min and 60-min coverage),
- sites that unlock **remote hinterlands** (large 120-min coverage), and
- how well these benefits align with the priority clusters.

> Flowchart below summarises the catchment analysis steps used to estimate people and cropland within 30/60/120 minutes of each site, and how these metrics feed into Table S3 and Figure S3.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S6_inputs["Inputs"]
        s6i1["Candidate sites and hubs"]
        s6i2["Travel time surface and network"]
        s6i3["Population grid"]
        s6i4["Cropland grid"]
        s6i5["Priority clusters and mask"]
  end
 subgraph S6_process["Processing"]
        s6p1["Build 30, 60, 120 minute isochrones per site"]
        s6p2["Compute people and cropland within each isochrone"]
        s6p3["Rank sites by 60 minute population coverage"]
        s6p4["Compute share of provincial population per site"]
        s6p5["Union catchments for top sites by threshold"]
        s6p6["Overlay unions with priority clusters and corridor"]
  end
 subgraph S6_outputs["Outputs for Summary 3"]
        s6o1["Table S3: top sites by 60 minute catchment"]
        s6o2["Figure S3: union catchments and clusters map"]
        s6o3["Messages on quick wins and remote hinterlands"]
  end
    s6i1 --> s6p1
    s6i2 --> s6p1
    s6i3 --> s6p2
    s6i4 --> s6p2
    s6i5 --> s6p6
    s6p1 --> s6p2
    s6p2 --> s6p3 & s6p4
    s6p3 --> s6p5 & s6o1
    s6p5 --> s6p6
    s6p6 --> s6o2
    s6o1 --> s6o3
    s6o2 --> s6o3
```

**Table S3** aggregates, for each province, the sites with the largest
**60-minute catchments**, showing:

- people and cropland within 60 minutes of each site, and
- the **share of the province’s population** that each site alone can reach
  within 60 minutes.

> **Table S3. Top sites by 60-minute catchment**  
> *Generated from `{AOI}_catchments_kpis.csv` and site metadata.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Province (AOI) | Site ID | Site type | Population within 60 min | Cropland (km²) within 60 min | % of provincial population within 60 min of this site |
| ---- | ---- | ---- | ---- | ---- | ---- | 
| Benguela       | site_24 | unknown   | 1654574   | 34.65    | 50.57  |
| Benguela       | site_1  | unknown   | 1643762   | 41.90    | 50.24  |
| Benguela       | site_19 | unknown   | 961441    | 147.34   | 29.39  |
| Benguela       | site_26 | unknown   | 393440    | 816.40   | 12.03  |
| Benguela       | site_25 | unknown   | 393440    | 816.40   | 12.03  |
| Benguela       | site_22 | unknown   | 240769    | 456.21   | 7.36   |
| Benguela       | site_21 | unknown   | 175871    | 502.51   | 5.38   |
| Benguela       | site_15 | unknown   | 167291    | 77.60    | 5.11   |
| Benguela       | site_20 | unknown   | 166425    | 296.00   | 5.09   |
| Benguela       | site_13 | unknown   | 134638    | 86.88    | 4.12   |

:::

:::{tab-item} Huambo
:sync: key2

| Province (AOI) | Site ID | Site type (e.g. market / hub) | Population within 60 min | Cropland (km²) within 60 min | % of provincial population within 60 min of this site |
| ---- | ---- | ---- | ---- | ---- | ---- |
| huambo   | site_1  | unknown   | 1367563  | 1320.54  | 45.63  |
| huambo   | site_2  | unknown   | 1355369  | 1186.10  | 45.22  |
| huambo   | site_20 | unknown   | 1338284  | 939.56   | 44.65  |
| huambo   | site_5  | unknown   | 1302686  | 937.28   | 43.46  |
| huambo   | site_6  | unknown   | 1280394  | 889.48   | 42.72  |
| huambo   | site_10 | unknown   | 1279382  | 743.51   | 42.69  |
| huambo   | site_12 | unknown   | 1273502  | 867.43   | 42.49  |
| huambo   | site_11 | unknown   | 1269161  | 819.31   | 42.34  |
| huambo   | site_13 | unknown   | 1269161  | 819.31   | 42.34  |
| huambo   | site_8  | unknown   | 1217688  | 761.90   | 40.63  |

:::

:::{tab-item} Bie
:sync: key3

:::

:::{tab-item} Moxico
:sync: key4

| Province (AOI) | Site ID | Site type (e.g. market / hub) | Population within 60 min | Cropland (km²) within 60 min | % of provincial population within 60 min of this site |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico | site_9  | unknown   | 480541  | 60.60  | 63.55 |
| Moxico | site_2  | unknown   | 68133   | 15.47  | 9.01  |
| Moxico | site_11 | unknown   | 47240   | 10.60  | 6.25  |
| Moxico | site_10 | unknown   | 47157   | 16.18  | 6.24  |
| Moxico | site_5  | unknown   | 19370   | 31.55  | 2.56  |
| Moxico | site_7  | unknown   | 14404   | 39.36  | 1.90  |
| Moxico | site_3  | unknown   | 9982    | 11.02  | 1.32  |
| Moxico | site_12 | unknown   | 7804    | 7.53   | 1.03  |
| Moxico | site_8  | unknown   | 3706    | 0.31   | 0.49  |
| Moxico | site_1  | unknown   | 1623    | 1.15   | 0.21  |

:::

:::{tab-item} Moxico Leste
:sync: key5

:::
::::

**Figure S3** illustrates, for a selected subset of sites along the corridor,
how the **30/60/120-minute catchments** nest within the priority clusters
and corridor infrastructure.

> **Figure S3. Catchment isochrones and priority clusters**  
> *Static map showing 30/60/120-minute union catchments around top-ranked
> sites, overlaid with priority clusters and corridor infrastructure.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S3-Benguela](../../outputs/figs/AGO_A4L_LC_Benguela_CatchmentPriorityAreas.png)
:::

:::{tab-item} Huambo
:sync: key2
![S3-Huambo](../../outputs/figs/AGO_A4L_LC_Huambo_CatchmentPriorityAreas.png)
:::

:::{tab-item} Bie
:sync: key3
![S3-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S3-Moxico](../../outputs/figs/AGO_A4L_LC_Moxico_CatchmentPriorityAreas.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S3-MoxicoLeste]()
:::
::::

---

## 4. Are we stacking with other investments or creating islands?

The synergies overlay (Step 13) quantifies, for each priority site and
cluster centroid, **how many projects** from Government, World Bank,
and other partners lie within specified radii (e.g. ≤5 km, ≤10 km, ≤30 km).

This allows us to identify:

- **high-opportunity nodes**, where multiple projects intersect within
  short distances (good candidates for bundling),
- **isolated clusters**, where there is strong need but few nearby projects,
  and
- places where an investment might help **bridge separate project islands**.

For the summary tables and maps in this section, we focus on a radius of
**30 km**, which is where meaningful co-location begins to appear along
the corridor. At tighter radii (≤10 km), many clusters currently have few
or no overlapping projects, indicating that synergies mostly emerge at
the wider corridor scale rather than right next to cluster centroids.

> Flowchart below shows how project locations from Government, the World Bank, and other partners are overlaid around priority clusters to construct the project density metrics presented in Table S4 and Figure S4. For provinces where synergies could not be computed, this flowchart represents the intended workflow.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S7_inputs["Inputs"]
        s7i1["Priority clusters and centroids"]
        s7i2["Government project locations"]
        s7i3["World Bank project locations"]
        s7i4["Other partners project locations"]
        s7i5["User chosen distance thresholds, for example 10 km or 30 km"]
  end
 subgraph S7_process["Processing"]
        s7p1["Count projects by type within each radius around clusters"]
        s7p2["Compute total projects per cluster and radius"]
        s7p3["Identify high opportunity nodes with many nearby projects"]
        s7p4["Flag isolated clusters with few projects nearby"]
        s7p5["Prepare summary table for chosen radius"]
  end
 subgraph S7_outputs["Outputs for Summary 4"]
        s7o1["Table S4: clusters with highest project density"]
        s7o2["Figure S4: map of cluster rings and nearby projects"]
        s7o3["Messages on bundling opportunities and gaps"]
  end
    s7i1 --> s7p1
    s7i2 --> s7p1
    s7i3 --> s7p1
    s7i4 --> s7p1
    s7i5 --> s7p1
    s7p1 --> s7p2
    s7p2 --> s7p3 & s7p4 & s7p5 & s7o2
    s7p5 --> s7o1
    s7o1 --> s7o3
    s7o2 --> s7o3
```

**Table S4** lists, for each province, the **clusters with the highest
number of nearby projects** within 30 km.

> **Table S4. Clusters with highest project density within 30 km**  
> *Generated from `{AOI}_cluster_synergies.csv`.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Province (AOI) | Cluster ID | Projects within 10 km (Gov) | Projects within 10 km (WB) | Projects within 10 km (Other) | Total projects within 10 km |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Benguela         | 3 | 0 | 3 | 0 | 3 |
| Benguela         | 2 | 0 | 2 | 0 | 2 |
| Benguela         | 4 | 0 | 2 | 0 | 2 |
| Benguela         | 1 | 0 | 1 | 0 | 1 |

:::

:::{tab-item} Huambo
:sync: key2

| Province (AOI) | Cluster ID | Projects within 10 km (Gov) | Projects within 10 km (WB) | Projects within 10 km (Other) | Total projects within 10 km |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Huambo         | 1 | 0 | 23 | 0 | 23 |
| Huambo         | 2 | 0 | 14 | 0 | 14 |

For Huambo, two priority clusters stand out when we look at projects
within a **30 km** radius:

- **Cluster 1** sits in a dense halo of World Bank operations, with
  **23 World Bank projects within 30 km** and no government or other-partner
  projects recorded in this radius.
- **Cluster 2** is also well connected, with **14 World Bank projects
  within 30 km**, again without overlapping government or other-partner
  investments in the same band.

At tighter distances (≤10 km), the synergies table reports **zero projects**
around both clusters. This suggests that, in Huambo, co-location with
other investments currently happens at the **corridor scale (≤30 km)** rather
than directly adjacent to the cluster centroids. New investments in these
clusters could therefore act as **anchors for bundling and coordination**
with existing World Bank portfolios, while still leaving room to attract
government and other partners closer to the priority nodes over time.

:::

:::{tab-item} Bie
:sync: key3

| Province (AOI) | Cluster ID | Projects within 10 km (Gov) | Projects within 10 km (WB) | Projects within 10 km (Other) | Total projects within 10 km |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Bie  | 1 | 0 | 0 | 0 | 0 |

:::

:::{tab-item} Moxico
:sync: key4

| Province (AOI) | Cluster ID | Projects within 10 km (Gov) | Projects within 10 km (WB) | Projects within 10 km (Other) | Total projects within 10 km |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico  |  |  |  |  |  |

:::

:::{tab-item} Moxico Leste
:sync: key5

| Province (AOI) | Cluster ID | Projects within 10 km (Gov) | Projects within 10 km (WB) | Projects within 10 km (Other) | Total projects within 10 km |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico Leste |  |  |  |  |  |

:::
::::

**Figure S4** shows a **corridor-wide map of project density**, where the
size of each cluster marker is proportional to the number of projects
within 30 km, and color encodes the dominant financier.

> **Figure S4. Project density around priority clusters (≤30 km)**  
> *Static map with cluster markers sized by project counts within 30 km,
> colored by dominant project type (Gov / WB / Other / Mixed).*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S4-Benguela](../../outputs/figs/benguela_fig_s4_cluster_synergies_30km.png)
:::

:::{tab-item} Huambo
:sync: key2
![S4-Huambo](../../outputs/figs/huambo_fig_s4_cluster_synergies_30km.png)
:::

:::{tab-item} Bie
:sync: key3
![S4-Bie](../../outputs/figs/bie_fig_s4_cluster_synergies_30km.png)
:::

:::{tab-item} Moxico
:sync: key4
![S4-Moxico](../../outputs/figs/moxico_fig_s4_cluster_synergies_30km.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S4-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s4_cluster_synergies_30km.png)
:::
::::

---

## 5. How does movement along the corridor reinforce these priorities?

Where OD-Lite has been run (Step 14), we have a simple gravity model of
flows between municipalities, using population and distances, optionally
tilted by wealth (RWI). This lens shows **which segments of the corridor
carry the most interaction**, and how that intersects with priorities.

We summarize:

- the **top OD pairs** by modelled flow,
- municipalities with the highest **combined in- and out-flows** (throughput),
- how many of those high-throughput municipalities intersect with
  **priority clusters and top-ranked Admin2s**, and
- whether **both ends of a flow** lie inside the **top priority mask**.

> Flowchart below summarises how the OD-Lite gravity model uses population, distances, and zone attributes to generate OD flows, and how these are aggregated into the high-throughput pairs shown in Table S5 and the OD map in Figure S5.

```{mermaid}
---
config:
  layout: elk
---
flowchart TB
 subgraph S8_inputs["Inputs"]
        s8i1["OD zones and centroids by Admin2"]
        s8i2["Population and optional RWI by zone"]
        s8i3["Inter zone distances"]
        s8i4["Priority mask and Admin2 priority flag"]
  end
 subgraph S8_process["Processing"]
        s8p1["Run gravity model to estimate OD flows"]
        s8p2["Symmetrise and aggregate flows to undirected pairs"]
        s8p3["Rank OD pairs by modelled flow"]
        s8p4["Compute throughput per municipality from in flows and out flows"]
        s8p5["Flag pairs where both ends are in top priority mask"]
  end
 subgraph S8_outputs["Outputs for Summary 5"]
        s8o1["Table S5: high throughput municipality pairs"]
        s8o2["Figure S5: OD flows overlaid on priority clusters"]
        s8o3["Messages on key movement corridors and priority alignment"]
  end
    s8i1 --> s8p1
    s8i2 --> s8p1
    s8i3 --> s8p1
    s8i4 --> s8p5
    s8p1 --> s8p2
    s8p2 --> s8p3 & s8p4
    s8p3 --> s8p5 & s8o1 & s8o2
    s8o1 --> s8o3
    s8o2 --> s8o3
```

> **Table S5. High-throughput municipality pairs**  
> *Generated from `{AOI}_od_gravity.csv` and `{AOI}_od_zone_attrs.csv`, including a
> flag indicating whether both origin and destination lie inside the top
> priority mask.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Benguela | Benguela   | Catumbela  | 103693.4 | 32.9  | No  |
| Benguela | Benguela   | Lobito     | 53129.5  | 80.8  | No  |
| Benguela | Benguela   | Cubal      | 45650.2  | 91.9  | No  |
| Benguela | Catumbela  | Lobito     | 41826.9  | 48.6  | No  |
| Benguela | Cubal      | Ganda      | 39421.1  | 41.9  | Yes |
| Benguela | Benguela   | Bocoio     | 27623.0  | 85.1  | No  |
| Benguela | Benguela   | Caimbambo  | 24313.9  | 51.7  | No  |
| Benguela | Bocoio     | Lobito     | 23924.1  | 49.8  | No  |
| Benguela | Benguela   | Ganda      | 20842.4  | 128.3 | No  |
| Benguela | Baia Farta | Benguela   | 20274.6  | 71.4  | No  |

:::

:::{tab-item} Huambo
:sync: key2

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Huambo | Caala    | Huambo              | 97938.8  | 55.7  | Yes |
| Huambo | Bailundo | Huambo              | 55540.2  | 96.8  | No  |
| Huambo | Huambo   | Tchikala-tcholoanga | 49511.4  | 35.9  | No  |
| Huambo | Huambo   | Katchiungo          | 41293.9  | 57.3  | No  |
| Huambo | Ekunha   | Huambo              | 33430.5  | 45.8  | No  |
| Huambo | Huambo   | Londuimbali         | 31016.6  | 84.9  | No  |
| Huambo | Huambo   | Longonjo            | 26294.1  | 68.8  | No  |
| Huambo | Bailundo | Mungo               | 18917.9  | 44.4  | No  |
| Huambo | Bailundo | Londuimbali         | 16082.6  | 68.6  | No  |
| Huambo | Huambo   | Ukuma               | 15144.4  | 70.5  | No  |

:::

:::{tab-item} Bie
:sync: key3

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Bie | Chinguar  | Kuito     | 75589.3  | 48.9  | No |
| Bie | Catabola  | Kuito     | 73571.0  | 50.4  | No |
| Bie | Camacupa  | Kuito     | 55258.8  | 83.2  | No |
| Bie | Cunhinga  | Kuito     | 35870.2  | 64.8  | No |
| Bie | Andulo    | Kuito     | 31944.5  | 153.1 | No |
| Bie | Andulo    | N'harea   | 28148.5  | 75.9  | No |
| Bie | Camacupa  | Catabola  | 19856.3  | 66.4  | No |
| Bie | Kuito     | N'harea   | 19714.8  | 137.6 | No |
| Bie | Chitembo  | Kuito     | 19496.8  | 106.6 | No |
| Bie | Andulo    | Cunhinga  | 14262.2  | 88.4  | No |

:::

:::{tab-item} Moxico
:sync: key4

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico | Leua           | Luena (moxico)     | 31106.8 | 108.5 | No |
| Moxico | Camanongue     | Luena (moxico)     | 29860.1 | 113.7 | No |
| Moxico | Camanongue     | Leua               | 6789.8  | 52.6  | No |
| Moxico | Luchazes       | Luena (moxico)     | 6200.2  | 160.6 | No |
| Moxico | Luena (moxico) | Lumbala N'guimbo   | 6010.6  | 267.8 | No |
| Moxico | Luchazes       | Lumbala N'guimbo   | 660.7   | 199.7 | No |
| Moxico | Leua           | Lumbala N'guimbo   | 417.4   | 285.8 | No |
| Moxico | Camanongue     | Lumbala N'guimbo   | 213.5   | 332.9 | No |
| Moxico | Leua           | Luchazes           | 146.6   | 250.4 | No |
| Moxico | Camanongue     | Luchazes           | 110.9   | 271.5 | No |

:::

:::{tab-item} Moxico Leste
:sync: key5

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Moxicoleste | Alto Zambeze  | Luau           | 53869.7 | 176.0 | No |
| Moxicoleste | Luacano       | Luau           | 38895.3 | 85.5  | No |
| Moxicoleste | Alto Zambeze  | Luacano        | 32368.1 | 111.3 | No |
| Moxicoleste | Kameia Lumege | Luau           | 27949.1 | 131.3 | No |
| Moxicoleste | Alto Zambeze  | Kameia Lumege  | 15639.8 | 183.5 | No |
| Moxicoleste | Kameia Lumege | Luacano        | 12327.9 | 87.2  | No |

:::
::::

> **Figure S5. Origin-Destination flows and priority clusters along the corridor**  
> *Static map showing thick OD arcs for the top flows, overlaid on the
> priority clusters and corridor infrastructure.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S5-Benguela](../../outputs/figs/benguela_fig_s5_od_flows.png)
:::

:::{tab-item} Huambo
:sync: key2
![S5-Huambo](../../outputs/figs/huambo_fig_s5_od_flows.png)
:::

:::{tab-item} Bie
:sync: key3
![S5-Bie](../../outputs/figs/bie_fig_s5_od_flows.png)
:::

:::{tab-item} Moxico
:sync: key4
![S5-Moxico](../../outputs/figs/moxico_fig_s5_od_flows.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S5-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s5_od_flows.png)
:::
::::

---

## 6. Are investments reaching the poorest?

The equity analysis (Step 09) goes beyond the quadrant scatter plot above
to produce a formal **benefit incidence curve** and **concentration index
(CI)** for each province. These tools answer a precise question: *when we
rank municipalities from poorest to richest, what share of priority-area
benefits accrues to the bottom half?*

> Flowchart below summarises how the benefit incidence curve and concentration index are derived from the municipality ranking, and how they feed into Table S6 and Figure S6.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S9_inputs["Inputs"]
        s9i1["Municipality composite scores (Step 09)"]
        s9i2["Rural poverty index per municipality"]
        s9i3["Population per municipality"]
        s9i4["Share of priority area per municipality"]
  end
 subgraph S9_process["Processing"]
        s9p1["Rank municipalities poorest to richest"]
        s9p2["Compute cumulative population share"]
        s9p3["Compute cumulative benefit share (priority area coverage)"]
        s9p4["Plot benefit incidence curve against 45 degree line"]
        s9p5["Calculate concentration index: CI = 2 times sum of gap times population shares"]
        s9p6["Classify: CI > 0 pro-poor, CI < 0 pro-rich, CI near 0 neutral"]
  end
 subgraph S9_outputs["Outputs for Summary 6"]
        s9o1["Table S6: concentration index by province"]
        s9o2["Figure S6: benefit incidence curves"]
        s9o3["Equity messages: pro-poor, neutral, or pro-rich"]
  end
    s9i1 --> s9p1
    s9i2 --> s9p1
    s9i3 --> s9p2
    s9i4 --> s9p3
    s9p1 --> s9p2 & s9p3
    s9p2 --> s9p4
    s9p3 --> s9p4
    s9p4 --> s9p5
    s9p5 --> s9p6
    s9p6 --> s9o1 & s9o3
    s9p4 --> s9o2
```

**Table S6** reports, for each province, the **concentration index** and
a plain-language interpretation. A positive CI means the current priority
surface disproportionately benefits poorer municipalities; a negative CI
means it skews toward better-off areas.

> **Table S6. Concentration index by province (illustrative structure)**
> *Generated from `{AOI}_equity_summary.csv` (one file per province).*

| Province (AOI) | Concentration Index (CI) | Interpretation | Number of municipalities in ranking | Share of benefits going to bottom 50% of population |
| ---- | ---- | ---- | ---- | ---- |
| Benguela | 0.5961 | pro-poor | 10 | 86.6% of benefits |
| Huambo | -0.3441 | pro-rich | 11 | 20.5% of benefits |
| Bie | -0.5464 | pro-rich | 9 | 0.0% |
| Moxico | -1.523 | pro-rich | 5 | 0.0% (no priority zones) |
| Moxico Leste | -1.3438 | pro-rich | 4 | 0.0% (no priority zones) |

**How to read the CI:**

- **CI > 0.05**: Priority selection is meaningfully **pro-poor** — the
  current weights and thresholds direct investment toward areas of greatest
  need.
- **CI near 0**: **Neutral** — the selection neither favors nor
  disadvantages poorer municipalities. Consider whether the policy intent
  calls for pro-poor targeting.
- **CI < −0.05**: **Pro-rich** — richer municipalities capture a
  disproportionate share of benefits. Review weights (especially
  `W_POV`, `W_FOOD`) or lower the minimum cropland threshold.

**Figure S6** shows, for each province, the **benefit incidence curve**.
If the curve lies above the 45-degree diagonal, priority-area benefits are
concentrated among poorer municipalities (pro-poor). If below, benefits
skew toward better-off areas.

> **Figure S6. Benefit incidence curves by province**
> *Generated from `{AOI}_benefit_incidence.csv`. The shaded area between
> the curve and the diagonal represents the concentration index.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S6-Benguela](../../outputs/figs/benguela_fig_s6_benefit_incidence.png)
:::

:::{tab-item} Huambo
:sync: key2
![S6-Huambo](../../outputs/figs/huambo_fig_s6_benefit_incidence.png)
:::

:::{tab-item} Bie
:sync: key3
![S6-Bie](../../outputs/figs/bie_fig_s6_benefit_incidence.png)
:::

:::{tab-item} Moxico
:sync: key4
![S6-Moxico](../../outputs/figs/moxico_fig_s6_benefit_incidence.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S6-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s6_benefit_incidence.png)
:::
::::

---

## 7. What if we upgrade the most vulnerable roads?

The intervention simulator (Step 16) models the impact of upgrading
flood-vulnerable road segments to secondary-road standard (45 km/h). It
answers: *how many people would gain faster access to project sites, and
by how much?*

> Flowchart below summarises how the intervention simulation upgrades flood-risk road cells, recomputes travel times, and produces the impact metrics presented in Table S7 and Figure S7.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S10_inputs["Inputs"]
        s10i1["Friction surface: minutes per km"]
        s10i2["Flood-risk road cells (Step 04)"]
        s10i3["Project site locations"]
        s10i4["Population grid"]
        s10i5["Target upgrade speed: 45 km/h"]
  end
 subgraph S10_process["Processing"]
        s10p1["Compute baseline travel time from all sites"]
        s10p2["Set flood-risk road cells to target speed"]
        s10p3["Recompute travel time on improved network"]
        s10p4["Subtract: minutes saved per grid cell"]
        s10p5["Count population by improvement threshold: 5, 10, 15, 30, 60 plus minutes"]
        s10p6["Count people newly within 60 and 120 minutes of a site"]
  end
 subgraph S10_outputs["Outputs for Summary 7"]
        s10o1["Table S7: population gaining by threshold"]
        s10o2["Figure S7: impact bar charts"]
        s10o3["Messages on quick wins and transformative gains"]
  end
    s10i1 --> s10p1
    s10i2 --> s10p2
    s10i3 --> s10p1
    s10i4 --> s10p5
    s10i5 --> s10p2
    s10p1 --> s10p4
    s10p2 --> s10p3
    s10p3 --> s10p4
    s10p4 --> s10p5 & s10p6
    s10p5 --> s10o1 & s10o2
    s10p6 --> s10o1
    s10o1 --> s10o3
    s10o2 --> s10o3
```

**Table S7** reports, for each province, the population gaining
at each improvement threshold after upgrading flood-risk road
cells to 45 km/h.

> **Table S7. Population gaining by improvement threshold**
> *Generated from `{AOI}_sim_impact_summary.csv` (one file per province).
> Scenario: `upgrade_risk_roads_to_45kmh`.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Threshold | Population gaining | Area gaining (km²) |
| ---- | ----: | ----: |
| ≥ 5 min saved | 987,712 | 31,590 |
| ≥ 10 min saved | 94,230 | 14,648 |
| ≥ 15 min saved | 18,963 | 5,096 |
| ≥ 30 min saved | 73 | 169 |
| ≥ 60 min saved | 42 | 62 |
| **Newly within 60 min** | **41,508** | **911** |
| **Newly within 120 min** | **26,320** | **969** |

:::

:::{tab-item} Huambo
:sync: key2

| Threshold | Population gaining | Area gaining (km²) |
| ---- | ----: | ----: |
| ≥ 5 min saved | 37,539 | 2,792 |
| ≥ 10 min saved | 17 | 2 |
| ≥ 15 min saved | 0 | 0 |
| ≥ 30 min saved | 0 | 0 |
| ≥ 60 min saved | 0 | 0 |
| **Newly within 60 min** | **6,143** | **150** |
| **Newly within 120 min** | **5,791** | **129** |

:::

:::{tab-item} Bie
:sync: key3

*Simulation data not yet available for Bie. Run Step 16 to generate.*

:::

:::{tab-item} Moxico
:sync: key4

| Threshold | Population gaining | Area gaining (km²) |
| ---- | ----: | ----: |
| ≥ 5 min saved | 73,049 | 188,626 |
| ≥ 10 min saved | 48,784 | 160,093 |
| ≥ 15 min saved | 32,172 | 140,848 |
| ≥ 30 min saved | 17,295 | 90,414 |
| ≥ 60 min saved | 4,596 | 41,873 |
| **Newly within 60 min** | **11,042** | **2,344** |
| **Newly within 120 min** | **5,015** | **6,623** |

:::

:::{tab-item} Moxico Leste
:sync: key5

*Simulation data not yet available for Moxico Leste. Run Step 16 to generate.*

:::
::::

**How to read the impact table:**

- **Large populations at low thresholds** (many people gaining 5–10 min):
  widespread but modest improvement — typical of upgrading a few key
  segments on a busy route.
- **Smaller populations at high thresholds** (fewer people gaining 30+ min):
  transformative improvement for remote communities — often more important
  for equity.
- **Newly within 60 min**: the single most actionable metric — these are
  people who previously had **no reasonable access** to project sites and
  now do.

**Figure S7** shows, for each province, a bar chart of population gaining
at each improvement threshold, with the "newly within 60 min" metric
highlighted.

> **Figure S7. Intervention impact: population gaining by threshold**
> *Generated from `{AOI}_sim_impact_summary.csv`. Bar chart showing
> population at each minutes-saved threshold, plus newly-within-60-min.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S7-Benguela](../../outputs/figs/benguela_fig_s7_intervention_impact.png)
:::

:::{tab-item} Huambo
:sync: key2
![S7-Huambo](../../outputs/figs/huambo_fig_s7_intervention_impact.png)
:::

:::{tab-item} Bie
:sync: key3
![S7-Bie](../../outputs/figs/bie_fig_s7_intervention_impact.png)
:::

:::{tab-item} Moxico
:sync: key4
![S7-Moxico](../../outputs/figs/moxico_fig_s7_intervention_impact.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S7-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s7_intervention_impact.png)
:::
::::

---

## 8. How many people does each site uniquely serve?

The marginal catchment analysis (Step 15) goes beyond raw catchment size
to answer: *of the population within 60 minutes of a site, how many are
**not** already closer to a higher-ranked site?* This measures the
**net-new beneficiaries** each project uniquely brings.

> Flowchart below summarises how the marginal catchment is computed by
> iteratively removing already-served populations, producing Table S8
> and Figure S8.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S8_inputs["Inputs"]
        s8i1["Isochrone catchments per site (Step 10)"]
        s8i2["Population grid"]
        s8i3["Site ranking by total 60-min catchment"]
  end
 subgraph S8_process["Processing"]
        s8p1["Rank sites by total catchment size (descending)"]
        s8p2["For top-ranked site: marginal = total catchment"]
        s8p3["For each subsequent site: subtract population already within a higher-ranked catchment"]
        s8p4["Compute pct_marginal = marginal / total"]
        s8p5["Repeat for 30, 60, 120, 240-min thresholds"]
  end
 subgraph S8_outputs["Outputs for Summary 8"]
        s8o1["Table S8: sites ranked by net-new beneficiaries"]
        s8o2["Figure S8: total vs marginal catchment per site"]
        s8o3["Messages on unique reach and overlap"]
  end
    s8i1 --> s8p1
    s8i2 --> s8p2
    s8i3 --> s8p1
    s8p1 --> s8p2
    s8p2 --> s8p3
    s8p3 --> s8p4
    s8p4 --> s8p5
    s8p5 --> s8o1 & s8o2
    s8o1 --> s8o3
```

**Table S8** reports, for each site in a province, the **net-new
population** at the 60-minute threshold. Sites with high total
catchment but low marginal share indicate significant overlap with
higher-ranked sites. Sites with high marginal share are uniquely
important.

> **Table S8. Net new beneficiaries by site (60-min threshold)**
> *Generated from `{AOI}_marginal_catchment.csv` (one file per province).
> Top 10 sites ranked by marginal population.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Site | Net New Pop | Total Catchment | % Marginal |
| ---- | ----: | ----: | ----: |
| S21 | 175,871 | 175,871 | 100.0% |
| S19 | 81,866 | 961,440 | 8.5% |
| S26 | 39,384 | 393,440 | 10.0% |
| S25 | 39,384 | 393,440 | 10.0% |
| S13 | 38,440 | 134,638 | 28.6% |
| S9 | 37,688 | 57,246 | 65.8% |
| S8 | 37,688 | 57,246 | 65.8% |
| S22 | 36,238 | 240,769 | 15.1% |
| S15 | 35,615 | 167,291 | 21.3% |
| S10 | 35,381 | 58,306 | 60.7% |

:::

:::{tab-item} Huambo
:sync: key2

| Site | Net New Pop | Total Catchment | % Marginal |
| ---- | ----: | ----: | ----: |
| S3 | 134,938 | 187,037 | 72.1% |
| S39 | 117,202 | 244,220 | 48.0% |
| S25 | 106,990 | 127,032 | 84.2% |
| S4 | 106,292 | 138,067 | 77.0% |
| S36 | 100,834 | 635,411 | 15.9% |
| S28 | 98,508 | 623,511 | 15.8% |
| S26 | 94,373 | 100,811 | 93.6% |
| S37 | 94,257 | 998,940 | 9.4% |
| S27 | 93,131 | 99,103 | 94.0% |
| S29 | 89,559 | 289,716 | 30.9% |

:::

:::{tab-item} Bie
:sync: key3

*Marginal catchment data not yet available for Bie. Run Step 15 to generate.*

:::

:::{tab-item} Moxico
:sync: key4

| Site | Net New Pop | Total Catchment | % Marginal |
| ---- | ----: | ----: | ----: |
| S5 | 19,370 | 19,370 | 100.0% |
| S7 | 10,912 | 14,404 | 75.8% |
| S3 | 9,982 | 9,982 | 100.0% |
| S12 | 7,576 | 7,804 | 97.1% |
| S10 | 6,146 | 47,157 | 13.0% |
| S8 | 3,706 | 3,706 | 100.0% |
| S1 | 1,623 | 1,623 | 100.0% |
| S11 | 1,500 | 47,240 | 3.2% |
| S4 | 1,472 | 1,472 | 100.0% |
| S2 | 842 | 68,133 | 1.2% |

:::

:::{tab-item} Moxico Leste
:sync: key5

*Marginal catchment data not yet available for Moxico Leste. Run Step 15 to generate.*

:::
::::

**How to read the marginal catchment:**

- **100% marginal**: This site serves an entirely new population that
  no higher-ranked site covers. Highest strategic value.
- **High total, low % marginal**: The site's catchment overlaps
  heavily with a higher-ranked site. Consider whether both sites are
  needed, or if resources should shift to less-served areas.
- **Small marginal, remote location**: Even modest net-new
  populations may represent important equity gains in underserved areas.

**Figure S8** shows, for each province, a stacked bar chart comparing
total catchment vs net-new (marginal) population per site at the
60-minute threshold. Gray bars show overlap with higher-ranked sites;
blue bars show unique reach.

> **Figure S8. Net new beneficiaries by site**
> *Generated from `{AOI}_marginal_catchment.csv`. Stacked horizontal
> bars showing overlap (gray) and net-new (blue) populations.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S8-Benguela](../../outputs/figs/benguela_fig_s8_marginal_catchment.png)
:::

:::{tab-item} Huambo
:sync: key2
![S8-Huambo](../../outputs/figs/huambo_fig_s8_marginal_catchment.png)
:::

:::{tab-item} Bie
:sync: key3
![S8-Bie](../../outputs/figs/bie_fig_s8_marginal_catchment.png)
:::

:::{tab-item} Moxico
:sync: key4
![S8-Moxico](../../outputs/figs/moxico_fig_s8_marginal_catchment.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S8-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s8_marginal_catchment.png)
:::
::::

---

## 9. Where are the road bottlenecks?

The OD-bottleneck overlay (Step 15) combines origin-destination flow
modelling with flood-risk mapping to pinpoint the road cells that
carry the highest gravity-weighted traffic **and** sit on
flood-vulnerable segments. These are the cells where a single washout
would disrupt the most market connections.

> Flowchart below summarises how OD flows are rasterised onto
> flood-risk road cells, producing Table S9 and Figure S9.

```{mermaid}
---
config:
  layout: dagre
---
flowchart TB
 subgraph S9b_inputs["Inputs"]
        s9bi1["OD flow matrix (Step 12)"]
        s9bi2["Flood-risk road cells (Step 04)"]
        s9bi3["Road network geometry"]
        s9bi4["Zone centroids"]
  end
 subgraph S9b_process["Processing"]
        s9bp1["Route each OD pair along shortest road path"]
        s9bp2["Accumulate gravity-weighted flow on each road cell"]
        s9bp3["Intersect flow grid with flood-risk mask"]
        s9bp4["Rank cells by flow_load (highest = most critical)"]
  end
 subgraph S9b_outputs["Outputs for Summary 9"]
        s9bo1["Table S9: top bottleneck cells ranked by flow load"]
        s9bo2["Figure S9: bottleneck map coloured by flow intensity"]
        s9bo3["Messages on critical infrastructure gaps"]
  end
    s9bi1 --> s9bp1
    s9bi2 --> s9bp3
    s9bi3 --> s9bp1
    s9bi4 --> s9bp1
    s9bp1 --> s9bp2
    s9bp2 --> s9bp3
    s9bp3 --> s9bp4
    s9bp4 --> s9bo1 & s9bo2
    s9bo1 --> s9bo3
```

**Table S9** lists the top 10 road bottleneck cells for each province,
ranked by `flow_load` (a gravity-weighted sum of trips passing through
the cell). Higher flow load means more people and goods depend on that
specific flood-vulnerable road segment.

> **Table S9. Road bottleneck ranking (top 10 cells)**
> *Generated from `{AOI}_od_bottleneck_cells.csv` (one file per province).*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

| Rank | Longitude | Latitude | Flow Load |
| ----: | ----: | ----: | ----: |
| 1 | 13.7042 | -12.5208 | 61,365 |
| 2 | 14.7208 | -13.1208 | 60,264 |
| 3 | 13.6458 | -12.6458 | 53,130 |
| 4 | 13.6708 | -12.6042 | 53,130 |
| 5 | 13.6458 | -12.6625 | 53,130 |
| 6 | 13.8042 | -12.2708 | 53,130 |
| 7 | 13.7958 | -12.2958 | 53,130 |
| 8 | 13.8208 | -12.8708 | 45,650 |
| 9 | 13.9625 | -12.9542 | 45,650 |
| 10 | 13.9875 | -12.9625 | 45,650 |

:::

:::{tab-item} Huambo
:sync: key2

| Rank | Longitude | Latitude | Flow Load |
| ----: | ----: | ----: | ----: |
| 1 | 15.7875 | -12.8208 | 370,099 |
| 2 | 15.7875 | -12.8292 | 97,939 |
| 3 | 15.7292 | -12.9375 | 97,939 |
| 4 | 15.7958 | -12.7375 | 55,540 |
| 5 | 15.7875 | -12.7708 | 55,540 |
| 6 | 15.8042 | -12.5542 | 55,540 |
| 7 | 15.7958 | -12.7458 | 55,540 |
| 8 | 15.7958 | -12.7542 | 55,540 |
| 9 | 15.8375 | -12.8458 | 49,511 |
| 10 | 16.0708 | -12.9542 | 49,511 |

:::

:::{tab-item} Bie
:sync: key3

| Rank | Longitude | Latitude | Flow Load |
| ----: | ----: | ----: | ----: |
| 1 | 17.6708 | -12.4625 | 62,704 |
| 2 | 17.5625 | -12.4792 | 55,259 |
| 3 | 17.2375 | -12.5125 | 55,259 |
| 4 | 16.5625 | -11.4958 | 46,207 |
| 5 | 16.9458 | -12.3708 | 35,870 |
| 6 | 16.5958 | -11.5708 | 31,944 |
| 7 | 16.9042 | -12.2958 | 31,944 |
| 8 | 16.9375 | -12.3708 | 31,944 |
| 9 | 17.3125 | -12.1792 | 19,856 |
| 10 | 17.4375 | -12.2542 | 19,856 |

:::

:::{tab-item} Moxico
:sync: key4

| Rank | Longitude | Latitude | Flow Load |
| ----: | ----: | ----: | ----: |
| 1 | 20.4125 | -11.8708 | 31,107 |
| 2 | 19.8208 | -12.2958 | 31,107 |
| 3 | 19.8292 | -12.2958 | 31,107 |
| 4 | 20.3958 | -11.8792 | 31,107 |
| 5 | 20.1375 | -11.6125 | 29,860 |
| 6 | 19.7958 | -12.2958 | 29,860 |
| 7 | 19.8792 | -12.1292 | 29,860 |
| 8 | 19.9708 | -11.9458 | 29,860 |
| 9 | 19.9958 | -11.9042 | 14,930 |
| 10 | 20.1625 | -11.5708 | 14,930 |

:::

:::{tab-item} Moxico Leste
:sync: key5

| Rank | Longitude | Latitude | Flow Load |
| ----: | ----: | ----: | ----: |
| 1 | 22.0042 | -10.9208 | 53,870 |
| 2 | 22.2458 | -11.2625 | 53,870 |
| 3 | 22.2542 | -11.2708 | 53,870 |
| 4 | 22.2292 | -11.2375 | 53,870 |
| 5 | 22.2375 | -11.2542 | 53,870 |
| 6 | 22.6625 | -11.8625 | 53,870 |
| 7 | 22.6708 | -11.8708 | 53,870 |
| 8 | 22.8125 | -12.0792 | 53,870 |
| 9 | 22.7875 | -12.0458 | 53,870 |
| 10 | 22.6458 | -11.8458 | 53,870 |

:::
::::

**How to read the bottleneck ranking:**

- **Top-ranked cells**: These are the single most critical road
  segments for market connectivity. A flood event here would disrupt
  the most OD pairs. Prioritize these for resilience investments
  (drainage, elevation, rerouting).
- **Clustered bottlenecks**: When several top cells cluster along
  the same road segment, the entire corridor section needs attention.
- **Flow load magnitude**: Compare across provinces to identify
  corridor-wide priorities. Higher absolute values indicate busier
  trade routes.

**Figure S9** maps the bottleneck cells for each province, with marker
size and colour indicating flow-load intensity. The background shows
priority clusters (green) and admin boundaries (gray) for context.

> **Figure S9. Road bottleneck map (flow load on flood-risk cells)**
> *Generated from `{AOI}_od_bottleneck_cells.csv`. Markers sized and
> coloured by flow load; top 3 cells annotated.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S9-Benguela](../../outputs/figs/benguela_fig_s9_bottleneck_map.png)
:::

:::{tab-item} Huambo
:sync: key2
![S9-Huambo](../../outputs/figs/huambo_fig_s9_bottleneck_map.png)
:::

:::{tab-item} Bie
:sync: key3
![S9-Bie](../../outputs/figs/bie_fig_s9_bottleneck_map.png)
:::

:::{tab-item} Moxico
:sync: key4
![S9-Moxico](../../outputs/figs/moxico_fig_s9_bottleneck_map.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S9-MoxicoLeste](../../outputs/figs/moxicoleste_fig_s9_bottleneck_map.png)
:::
::::

---

## 10. How to read this page alongside the rest of the book

- This **Summary & results** page is a **corridor-wide dashboard**:
  it aligns indicators across provinces and views.
- **Per-view details** (what each map, table, and indicator really means)
  live in:
  - Chapters 1–10 (decision views, including bottlenecks and impact simulation),
  - Chapters 11–17 (run anywhere, data, methods, appendix).
- The **underlying code** that produced every table and map here is fully
  documented in:
  - [How it works](12-how-it-works.md),
  - [All pipeline code](../references/all-code.md).

Future runs for additional provinces will simply add rows to Tables S1–S9
and additional tabs to Figures S1–S9, without changing the
underlying methodology. The **corridor dashboard** (Step 15) automates
this aggregation.

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

![flow-top](../../outputs/figs/summary_all.png)

---

## 1. Where do the strongest priority clusters emerge?

Across all provinces processed so far, the 1-km priority surface (Step 07)
reveals a small number of **high-intensity clusters** that concentrate
most of the potential beneficiaries. These clusters represent places where:

- multiple constraints stack (low baseline access, poor electrification,
  high rural poverty and/or food insecurity), and
- there is enough **population and cropland** to justify coordinated
  investments.

> Figure S1-flow provides a schematic of how the 1-km priority surface, population, and cropland data are combined to identify priority clusters and produce Table S1 and Figure S1.

![flow-s1](../../outputs/figs/summary_1.png)

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
| Benguela | … | … | … | … | … | … | … | … | … | … |
| Huambo | 2 | 52.86 | 31776.0 | 2997196.75 | 1.06 | 34.72 | 10156.91 | 0.34 | 601.14 | 0.66 |
| Bie      | …  | … | … | … | … | … | … | … | … | … |
| Moxico | 0  | 0.0  | 0.0  | 756128.37  | 0.0 | 0.0 | 346.98 | 0.0 | NaN | NaN |
| Moxico Leste | … | …  | …  | …  | … | … | … | … | … | … |

**Figure S1** shows the **spatial pattern of priority clusters** along the
corridor: clusters hug the rail and primary road spine in a few key
segments, rather than being evenly spread across all municipalities.

> **Figure S1. Corridor-wide priority clusters**  
> *Static map showing priority clusters (Step 11) overlaid on the
> Lobito Corridor rail line and key roads, with provincial boundaries.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S1-Benguela]()
:::

:::{tab-item} Huambo
:sync: key2
![S1-Huambo](../../outputs/figs/huambo_fig_s1_priority_clusters_adm2_roads.png)
:::

:::{tab-item} Bie
:sync: key3
![S1-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S1-Moxico](../../outputs/figs/moxico_fig_s1_priority_clusters_adm2_roads.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S1-MoxicoLeste]()
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

> Figure S2-flow outlines how municipality-level indicators are normalised and combined into a composite score, and how this links to the quadrant analysis of priority score versus rural poverty used for Table S2 and Figure S2.

![flow-s2](../../outputs/figs/summary_2.png)

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
| Benguela | ... | ... | ... | ... | ... | ... | ... |
| Benguela | ... | ... | ... | ... | ... | ... | ... |
| Benguela | ... | ... | ... | ... | ... | ... | ... |
| Benguela | ... | ... | ... | ... | ... | ... | ... |
| Benguela | ... | ... | ... | ... | ... | ... | ... |

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
| Bie | ... | ... | ... | ... | ... | ... | ... |
| Bie | ... | ... | ... | ... | ... | ... | ... |
| Bie | ... | ... | ... | ... | ... | ... | ... |
| Bie | ... | ... | ... | ... | ... | ... | ... |
| Bie | ... | ... | ... | ... | ... | ... | ... |

:::

:::{tab-item} Moxico
:sync: key4

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico | Luena (moxico)        | 0.748  | 0.674  | 0.640 | 145.567  | 4.9  | NaN |
| Moxico | Lumbala N'guimbo      | 0.549  | 0.740  | 0.684 | 484.487  | 1.6  | NaN |
| Moxico | Luchazes              | 0.434  | 0.675  | 0.693 | 327.545  | 5.1  | NaN |
| Moxico | Camanongue            | NaN    | 0.728  | 0.738 | 101.129  | 4.5  | NaN |
| Moxico | Leua                  | NaN    | 0.627  | 0.784 | 75.766   | 18.5 | NaN |

:::

:::{tab-item} Moxico Leste
:sync: key5

| Province (AOI) | Municipality (Admin2) | Composite score (0–1) | Rural poverty index | Food insecurity index | Mean travel time (min) | % electrified | Share of province priority area (%) |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Moxico Leste | ... | ... | ... | ... | ... | ... | ... |
| Moxico Leste | ... | ... | ... | ... | ... | ... | ... |
| Moxico Leste | ... | ... | ... | ... | ... | ... | ... |
| Moxico Leste | ... | ... | ... | ... | ... | ... | ... |
| Moxico Leste | ... | ... | ... | ... | ... | ... | ... |

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
![S2-Benguela]()
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
![S2-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S2-Moxico](../../outputs/figs/moxico_fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| Quadrant (score x poverty)  | Number of municipalities | Estimated rural poor (people) | Share of province’s rural poor (%) |
| ---- | ---- | ---- | ---- |
| High score & high poverty   | 3  | 1532.0  | 26.0  |
| Lower score & lower poverty | 2  | 4467.0  | 74.0  |

- About 3 of 5 municipalities (60.0%) fall in the **high score & high poverty** quadrant, representing roughly 26.0% of the estimated rural poor in Moxico covered by the dataset.
:::

:::{tab-item} Moxico Leste
:sync: key5
![S2-MoxicoLeste]()
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

> Figure S3-flow summarises the catchment analysis steps used to estimate people and cropland within 30/60/120 minutes of each site, and how these metrics feed into Table S3 and Figure S3.

![flow-s3](../../outputs/figs/summary_3.png)

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
![S3-Benguela]()
:::

:::{tab-item} Huambo
:sync: key2
![S3-Huambo](../../outputs/figs/huambo_fig_s3_catchments_union.png)
:::

:::{tab-item} Bie
:sync: key3
![S3-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S3-Moxico](../../outputs/figs/moxico_fig_s3_catchments_union.png)
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

> Figure S4-flow shows how project locations from Government, the World Bank, and other partners are overlaid around priority clusters to construct the project density metrics presented in Table S4 and Figure S4. For provinces where synergies could not be computed, this flowchart represents the intended workflow.

![flow-s4](../../outputs/figs/summary_4.png)

**Table S4** lists, for each province, the **clusters with the highest
number of nearby projects** within 30 km.

> **Table S4. Clusters with highest project density within 30 km**  
> *Generated from `{AOI}_cluster_synergies.csv`.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

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

:::

:::{tab-item} Moxico
:sync: key4

:::

:::{tab-item} Moxico Leste
:sync: key5

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
![S4-Benguela]()
:::

:::{tab-item} Huambo
:sync: key2
![S4-Huambo](../../outputs/figs/huambo_fig_s4_cluster_synergies_30km.png)
:::

:::{tab-item} Bie
:sync: key3
![S4-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S4-Moxico]()
:::

:::{tab-item} Moxico Leste
:sync: key5
![S4-MoxicoLeste]()
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

> Figure S5-flow summarises how the OD-Lite gravity model uses population, distances, and zone attributes to generate OD flows, and how these are aggregated into the high-throughput pairs shown in Table S5 and the OD map in Figure S5.

![flow-s5](../../outputs/figs/summary_5.png)

> **Table S5. High-throughput municipality pairs**  
> *Generated from `{AOI}_od_gravity.csv` and `{AOI}_od_zone_attrs.csv`, including a
> flag indicating whether both origin and destination lie inside the top
> priority mask.*

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Huambo | Caala    | Huambo              | 97938.8  | 55.7  | Yes  |
| Huambo | Bailundo | Huambo              | 55540.2  | 96.8  | No  |
| Huambo | Huambo   | Tchikala-tcholoanga | 49511.4  | 35.9  | No  |
| Huambo | Huambo   | Katchiungo          | 41293.9  | 57.3  | No  |
| Huambo | Ekunha   | Huambo              | 33430.5  | 45.8  | No  |
| Huambo | Huambo   | Londuimbali         | 31016.6  | 84.9  | No  |
| Huambo | Huambo   | Longonjo            | 26294.1  | 68.8  | No  |
| Huambo | Bailundo | Mungo               | 18917.9  | 44.4  | No  |
| Huambo | Bailundo | Londuimbali         | 16082.6  | 68.6  | No  |
| Huambo | Huambo   | Ukuma               | 15144.4  | 70.5  | No  |

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

:::

:::{tab-item} Huambo
:sync: key2

:::

:::{tab-item} Bie
:sync: key3

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

:::
::::

> **Figure S5. Origin-Destination flows and priority clusters along the corridor**  
> *Static map showing thick OD arcs for the top flows, overlaid on the
> priority clusters and corridor infrastructure.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1
![S5-Benguela]()
:::

:::{tab-item} Huambo
:sync: key2
![S5-Huambo](../../outputs/figs/huambo_fig_s5_od_flows.png)
:::

:::{tab-item} Bie
:sync: key3
![S5-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S5-Moxico](../../outputs/figs/moxico_fig_s5_od_flows.png)
:::

:::{tab-item} Moxico Leste
:sync: key5
![S5-MoxicoLeste]()
:::
::::

---

## 6. How to read this page alongside the rest of the book

- This **Summary & results** page is a **corridor-wide dashboard**:
  it aligns indicators across provinces and views.
- **Per-view details** (what each map, table, and indicator really means)
  live in:
  - Chapters 1–8 (decision views),
  - Chapters 9–15 (run anywhere, data, methods).
- The **underlying code** that produced every table and map here is fully
  documented in:
  - [How it works](12-how-it-works.md),
  - [All pipeline code](../references/all-code.md).

Future runs for additional provinces will simply add rows to Tables S1–S5
and additional curves/markers to Figures S2–S5, without changing the
underlying methodology.

# Corridor-wide results & summary

This page summarizes the **main quantitative results** from the corridor
analysis. It is designed for decision makers and analysts who want to see
the *numbers, charts and maps* before diving into the detailed views in
Chapters 1–8.

The narrative here is **province-agnostic**: the same indicators are
reported for each AOI where the pipeline has been run (e.g. Huambo now,
additional provinces later). Differences across provinces come entirely
from the data, not from ad hoc methods.

---

## 1. Where do the strongest priority clusters emerge?

Across all provinces processed so far, the 1-km priority surface (Step 07)
reveals a small number of **high-intensity clusters** that concentrate
most of the potential beneficiaries. These clusters represent places where:

- multiple constraints stack (low baseline access, poor electrification,
  high rural poverty and/or food insecurity), and
- there is enough **population and cropland** to justify coordinated
  investments.

**Table S1** below summarizes, for each province and its top clusters:

- the **total priority area** in km²,
- the share of the province’s population located inside the priority mask,
- the share of cropland inside the priority mask, and
- the number of distinct clusters.

> **Table S1. Priority clusters by province (illustrative structure)**  
> *To be generated from `{AOI}_priority_clusters.csv` and
> `{AOI}_priority_top10_mask.tif`.*

| Province (AOI) | No. of clusters | Priority area (km²) | % of province population in priority mask | % of cropland in priority mask |
| ---- | ---- | ---- | ---- | ----|
| Benguela | …  | …      | …     | …     |
| Huambo | 2  | 52.86  | 1.06  | 0.34  |
| Bie      | …  | …      | …     | …     |
| Moxico | …  | …      | …     | …     |
| Moxico Leste | …  | …      | …     | …     |

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
![S1-Huambo](../../outputs/figs/fig_s1_priority_clusters_huambo_adm2_roads.png)
:::

:::{tab-item} Bie
:sync: key3
![S1-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S1-Moxico]()
:::

:::{tab-item} Moxico Leste
:sync: key5
![S1-MoxicoLeste]()
:::
::::

---

## 2. Are we focusing where needs and opportunities coincide?

The Admin2 analysis deliberately balances **need** and **opportunity**
when scoring municipalities (Step 09).

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
| Huambo | Longonjo | 0.549 | 0.616 | 0.730 | 286.90 | 0.005 | 0.00 |
| Huambo | Caala | 0.509 | 0.635 | 0.663 | 71.650 | 0.024 | 49.16 |
| Huambo | Mungo | 0.469 | 0.673 | 0.692 | 182.182 | 0.004 | 0.00 |
| Huambo | Londuimbali | 0.463 | 0.650 | 0.644 | 351.733 | 0.009 | 0.00 |
| Huambo | Ekunha | 0.394 | 0.683 | 0.685 | 61.163 | 0.006 | 0.00 |

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
| Moxico | ... | ... | ... | ... | ... | ... | ... |
| Moxico | ... | ... | ... | ... | ... | ... | ... |
| Moxico | ... | ... | ... | ... | ... | ... | ... |
| Moxico | ... | ... | ... | ... | ... | ... | ... |
| Moxico | ... | ... | ... | ... | ... | ... | ... |

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
![S2-Huambo](../../outputs/figs/fig_s2_priority_vs_poverty_scatter.png)

Quadrant Statistics

| quadrant | n_munis | rural_poor_est | share_corridor_rural_poor_pct |
| ---- | ---- | ---- | ---- |
| 0 - High score & high poverty   | 4  | 434986.119697  | 24.450089  |
| 1 - High score & lower poverty  | 2  | 342829.078235  | 19.270044  |
| 2 - Lower score & high poverty  | 2  | 146194.196262  | 8.217414   |
| 3 - Lower score & lower poverty | 3  | 855068.445733  | 48.062453  |

- About 4 of 11 municipalities (36.4%) fall in the **high score & high poverty** quadrant, representing roughly 24.5% of the estimated rural poor covered by the dataset.
- Around 2 municipalities (18.2%) sit in the **lower score & high poverty** quadrant, accounting for about 8.2% of the estimated rural poor — these are potentially under-prioritized areas.

:::

:::{tab-item} Bie
:sync: key3
![S2-Bie]()
:::

:::{tab-item} Moxico
:sync: key4
![S2-Moxico]()
:::

:::{tab-item} Moxico Leste
:sync: key5
![S2-Moxico Leste]()
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

| Province (AOI) | Site ID | Site type (e.g. market / hub) | Population within 60 min | Cropland (km²) within 60 min | % of provincial population within 60 min of any site |
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

:::

:::{tab-item} Huambo
:sync: key2
[S3-Huambo](../../outputs/figs/huambo_fig_s3_catchments_union.png)
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

**Table S4** lists, for each province, the **clusters with the highest
number of nearby projects** in a given radius.

> **Table S4. Clusters with highest project density (illustrative structure)**  
> *To be generated from `{AOI}_cluster_synergies.csv`.*

::::{tab-set}
:::{tab-item} Benguela
:sync: key1

:::

:::{tab-item} Huambo
:sync: key2

| Province (AOI) | Cluster ID | Projects within 10 km (Gov) | Projects within 10 km (WB) | Projects within 10 km (Other) | Total projects within 10 km |
| ---- | ---- | ---- | ---- | ---- | ---- |
| Huambo         | … | … | … | … | … |
| …              | … | … | … | … | … |

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
within 10 km, and color encodes the dominant financier.

> **Figure S4. Project density around priority clusters**  
> *Static map with cluster markers sized by project counts within 10 km,
> colored by dominant project type (Gov / WB / Other).*


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

:::

:::{tab-item} Moxico Leste
:sync: key5

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
  **priority clusters and top-ranked Admin2s**.

> **Table S5. High-throughput municipality pairs (illustrative structure)**  
> *To be generated from `{AOI}_od_gravity.csv` and `{AOI}_od_zone_attrs.csv`.*

| Province (AOI) | Origin Admin2 | Destination Admin2 | Modelled flow (relative) | Distance (km) | Both in top priority mask? |
| -------------- | ------------- | ------------------ | ------------------------ | ------------- | --------------------------- |
| Huambo         | …             | …                  | …                        | …             | Yes / No                    |
| …              | …             | …                  | …                        | …             | …                           |

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

:::

:::{tab-item} Moxico Leste
:sync: key5

:::
::::

> **Figure S5. OD flows and priority clusters along the corridor**  
> *Static map showing thick OD arcs for the top flows, overlaid on the
> priority clusters and corridor infrastructure.*

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

:::

:::{tab-item} Moxico Leste
:sync: key5

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

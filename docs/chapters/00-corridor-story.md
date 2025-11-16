# 0. Corridor story & key findings

This page connects the **five questions** from the introduction to the concrete
views in this book. It is written for task team leaders, government counterparts,
and analysts who want to see *how the pieces fit together* along the Lobito
Corridor.

Each subsection below answers one question from [the introduction](../index.md),
with pointers to the relevant chapters, tables, charts, and maps.

---

## 1. Impact: who benefits if we start in these places?

**Question from the intro:**  
> *How many households benefit from each option? Where are the poorest? Which areas maximize cropland and access improvements?*

The pipeline first builds a **priority surface** at 1-km resolution (Step 07),
then aggregates it to:

- **Priority clusters** (Step 11) — contiguous hotspots in the 1-km grid  
  → `outputs/tables/{AOI}_priority_clusters.csv`  
- **Municipality scores** (Steps 07 + 09) — Admin2 shortlist with equity and
  access indicators  
  → `outputs/tables/{AOI}_priority_muni_rank.csv`

From these two outputs we can tell a concise story:

- A **small number of clusters** along the Lobito–Huambo–Caála–Luena–Luau axis
  concentrate a large share of potential beneficiaries.
- Within those clusters, some municipalities combine **high rural poverty**,
  **significant cropland**, and **poor baseline access**, making them strong
  candidates for Diversifica Mais–type investments.

**What to look at**

- **Chapter 1 – Decide first**: top 3 clusters and top municipalities,
  including:
  - population in the priority mask,
  - cropland area,
  - share of electrified population,
  - average travel time to markets.
- **Chapter 2 – Municipality shortlist**: full Admin2 ranking with
  rural poverty, food insecurity, and access metrics.

**Suggested page elements**

- *Table 1*: **Top 3–5 priority clusters**, with columns:
  `cluster_id`, `NAM_1`, `NAM_2`, `pop`, `cropland_km2`,
  `%_electrified`, `mean_travel_min`.
- *Table 2*: **Top 10 municipalities by composite score**, from
  `priority_muni_rank.csv`.
- *Map 1*: Corridor map showing the **priority surface** and **cluster polygons**
  overlaid on the rail/road spine.

---

## 2. Spatial coordination: are we reinforcing or scattering investments?

**Question from the intro:**  
> *Are projects clustered to reinforce each other or spread too thin?*

The cluster view (Step 11) and municipal shortlist (Step 09) show how
priorities “clump” spatially:

- Clusters are enforced to meet a **minimum number of cells**
  (`MIN_CLUSTER_CELLS`), so we avoid isolated speckles.
- Municipalities are scored not only on their internal priority area, but also
  on **how much of the corridor’s priority mask they contain**.

Together, these outputs tell us whether:

- the corridor strategy is **concentrated in a few strong corridors**,
  or
- **thinly spread** across many municipalities with small, isolated patches.

**What to look at**

- **Chapter 3 – Actionable hotspots**:
  - number and size of clusters per province,
  - share of corridor-wide priority area in each cluster.
- **Chapter 2 – Municipality shortlist**:
  - column for `%_priority_area_in_muni`.

**Suggested page elements**

- *Map 2*: clusters overlaid on Admin2 boundaries, with the **Lobito rail**
  and key roads; high-priority clusters should visually “hug” the corridor in
  a few segments rather than everywhere.
- *Small chart*: bar plot showing **cluster area vs. population** to highlight
  whether we are putting more area into low-density or higher-density segments.

---

## 3. Equity: do priorities align with rural poverty & food insecurity?

**Question from the intro:**  
> *Do investment priorities align with rural poverty and food insecurity?*

Equity is handled at the **Admin2 level** using RAPP themes (Step 06) and the
composite targeting table (Step 09):

- rural poverty and food insecurity scores are brought in from RAPP,
- the composite score can optionally tilt via `W_POV`, `W_FOOD`, `W_RWI`,
- we can test how priority rankings change if we *remove* those weights.

**What to look at**

- **Chapter 7 – Equity lens**:
  - scatter plots of **priority score vs. rural poverty**,
  - scatter plots of **priority score vs. food insecurity**,
  - lists of **“high poverty, low priority”** municipalities (possible blind spots),
  - lists of **“low poverty, high priority”** municipalities (strategic trade-offs).

**Suggested page elements**

- *Chart*: 2×2 grid of scatter plots
  (priority vs. rural poverty; priority vs. food insecurity),
  with notable outliers labelled by municipality name.
- *Table 3*: list of the **top 5 “missed” poor municipalities** where rural
  poverty and food insecurity are high, but current priority is relatively low.

---

## 4. Synergies: can we stack with other investments?

**Question from the intro:**  
> *How close are proposed projects to existing World Bank, government, and other investments? Where can we coordinate?*

Synergies are quantified in **Step 13 – Synergies overlay**:

- For each project site or cluster centroid we compute:
  - distance to the nearest Government, World Bank, and “Other” project, and
  - counts of projects within several radii (e.g. ≤5 km, ≤10 km, ≤30 km)  
    → `site_synergies.csv`, `cluster_synergies.csv`.

This allows us to:

- identify **sites and clusters with the highest coordination potential**, and
- flag **“islands”** with virtually no nearby projects (which may need a
  stronger justification or different financing instruments).

**What to look at**

- **Chapter 6 – Synergies**:
  - top clusters by number of nearby projects,
  - sites with *zero* projects within 10–30 km,
  - municipality summaries of project density.

**Suggested page elements**

- *Map 3*: clusters and sites with graduated symbols for **number of
  projects within 10 km**, colored by project owner (Gov/WB/Other).
- *Table 4*: top 10 **clusters by total projects within 10 km** and their
  main municipalities.

---

## 5. Logistics: do upgrades improve market access along the rail & road spine?

**Question from the intro:**  
> *Do planned upgrades improve market and finance access while supporting the Caála Logistics Platform and other key nodes?*

Two components speak to logistics:

1. **Catchments** (Step 12):  
   travel-time catchments from project sites (e.g. 30/60/120 minutes) with
   population and cropland coverage  
   → `catchments_kpis.csv`.

2. **OD flows** (Step 14 – OD-Lite):  
   a simple gravity model between municipalities, using population and
   distance, plus optional RWI tilting  
   → `od_gravity.csv`, `od_zone_attrs.csv`, `od_agents.csv`.

Together, they tell us:

- which **sites serve the most people within a given time band**,
- which sites **“unlock” access to Caála and other key logistics nodes**, and
- which Admin2s sit on **high-throughput flows** (and may need resilient
  infrastructure).

**What to look at**

- **Chapter 5 – Catchments**:
  - top sites by population within 60 minutes,
  - trade-offs between reaching more people vs. more cropland.
- **Chapter 8 – Movement**:
  - top OD pairs by modelled flow,
  - municipalities with the highest combined in- and out-flows.

**Suggested page elements**

- *Map 4*: catchment isochrones (30/60/120 minutes) around a short list of
  candidate sites along the corridor (including Caála).
- *Chart*: bar chart of **population served within 60 minutes** for the top
  5–10 sites.

---

## 6. How to use this page

- **Decision makers** can use this page as a *one-stop briefing*:
  high-level findings, key maps, and the most important tables.
- **Analysts** can jump from here into the detailed chapters:
  - cluster-level analysis (Ch. 3),
  - full municipal targeting (Ch. 2 & 7),
  - catchments (Ch. 5),
  - synergies (Ch. 6),
  - movement (Ch. 8).

Under the hood, everything is fully reproducible via the
[Run anywhere](../chapters/09-cheatsheet.md) and
[How it works](../chapters/12-how-it-works.md) chapters. This page is the
storytelling layer sitting on top of the 1-km grid, cluster KPIs, and
municipal targeting tables generated by Steps 00–14.

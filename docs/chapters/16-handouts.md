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
# 16. Visual handouts

Two printable pages that summarise the analytical framework and pipeline at a glance.
Open the SVG links for full resolution (scales to any size), or use the PNG versions for quick embedding.

---

## Page 1 — Framework at a Glance

A single-page summary of the five strategic questions the tool answers,
populated with key findings from the **Balanced** scenario across all five provinces.

```{figure} ../deck/Lobito_Corridor_Framework_Page1.svg
:name: fig-framework-at-a-glance
:alt: Lobito Corridor analytical framework at a glance — 5 questions, key findings, and 4 extended analytics
:width: 100%

**Framework at a Glance.** Five configurable questions (priority clusters, municipal alignment, catchment beneficiaries, project synergies, economic flows) with key numbers from the Balanced scenario, plus four extended analytics layers (equity, net new beneficiaries, bottlenecks, what-if simulator).
```

````{dropdown} What does this page show?
:open:

| Section | Content |
|---------|---------|
| **Q1 — Priority Clusters** | Number of clusters, area, and population per province under the current threshold |
| **Q2 — Municipal Priorities** | Composite-score ranking with quadrant breakdown (invest / at-risk / growth / lower) |
| **Q3 — Catchment Beneficiaries** | Population within 30 / 60 / 120 min of top sites |
| **Q4 — Project Synergies** | WB / Gov / Other project density near priority clusters |
| **Q5 — Economic Flows** | Top OD pair and whether both ends fall in the priority mask |
| **Scenario Dial** | Equity-Heavy / Balanced / Growth-Heavy — all parameters adjustable in `config.py` |
| **Extended Analytics** | Concentration Index, marginal catchment, road bottlenecks, intervention simulator |
````

---

## Page 2 — How the Analysis Works

A pipeline flowchart showing all 17 processing steps (Steps 00-16), grouped into
three phases, with the configurable dials and a question-to-step mapping.

```{figure} ../deck/Lobito_Corridor_Framework_Page2.svg
:name: fig-pipeline-overview
:alt: Lobito Corridor pipeline overview — data inputs, 3 processing phases, and outputs
:width: 100%

**How the Analysis Works.** Data inputs (raster, vector, survey) flow through three phases: Foundation (grid alignment, isochrones, flood risk, RAPP ingest), Core Analysis (priority surface, municipality targeting, clusters, catchments, synergies), and Extended Analytics (equity, net-new beneficiaries, OD gravity, what-if simulator, corridor dashboard). All weights, thresholds, and component toggles are configurable.
```

````{dropdown} Pipeline phases at a glance
:open:

| Phase | Steps | What happens |
|-------|-------|-------------|
| **1 — Foundation** | 00, 01, 02, 04, 05, 06 | Align all layers to a common 1-km grid; build isochrone masks; flag flood-risk road cells; ingest socioeconomic indicators |
| **2 — Core Analysis** | 07, 08, 09, 10, 11, 12, 13 | Build the configurable priority surface; rank municipalities; identify clusters; compute travel-time catchments; overlay project synergies |
| **3 — Extended** | 09b, 12b, 14, 15, 16 | Equity measurement (CI, benefit incidence); marginal catchment (net-new); OD gravity model + bottleneck ranking; what-if road upgrade simulator; corridor-wide dashboard |
````

---

## Downloads

| File | Format | Description |
|------|--------|-------------|
| {download}`Framework at a Glance (SVG) <../deck/Lobito_Corridor_Framework_Page1.svg>` | SVG | Scalable vector — best for printing at any size |
| {download}`Framework at a Glance (PNG) <../deck/Lobito_Corridor_Framework_Page1.png>` | PNG | Raster — for quick embedding in slides or emails |
| {download}`Pipeline Overview (SVG) <../deck/Lobito_Corridor_Framework_Page2.svg>` | SVG | Scalable vector — best for printing at any size |
| {download}`Pipeline Overview (PNG) <../deck/Lobito_Corridor_Framework_Page2.png>` | PNG | Raster — for quick embedding in slides or emails |

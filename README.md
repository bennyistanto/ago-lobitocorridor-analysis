# Lobito Corridor Spatial Analysis

> **Project Reference:** Diversifica Mais ([P178035](https://opswork.worldbank.org/home/P178035))

This project helps prioritize last-mile infrastructure investments in Angolan provinces along the [Lobito Corridor](https://www.lobitocorridor.org/history-background). We combine Earth Observation data and public geospatial datasets with local socio-economic statistics to identify where limited resources can have the biggest impact.

![Lobito-Corridor](./docs/images/AGO_A4L_LC_Admin.png)
*Figure: Lobito Corridor alignment and intersecting provinces*

## Why the Lobito Corridor matters

The Lobito Corridor connects inland production areas to Atlantic export ports, running through provinces with high rural poverty, limited services, and seasonal access problems. Smart investments in roads, bridges, water, and power infrastructure can unlock agricultural value chains, reduce travel times to markets, and boost returns from larger corridor projects.

## The questions we help answer (from the team)

Our team uses this analysis to answer:

1. **Impact assessment**: How many households benefit from each project, especially the poorest? Which areas maximize cropland and access improvements?

2. **Spatial coordination**: Are projects clustered to reinforce each other or spread too thin?

3. **Equity**: Do investment priorities align with rural poverty and food insecurity patterns?

4. **Synergies**: How close are proposed projects to existing World Bank, government, and other investments? Where can we coordinate?

5. **Logistics**: Do planned upgrades improve market and finance access while supporting the Caála Logistics Platform and other key nodes?

## Data (lean by design)

We keep the data requirements lean:

**Earth Observation and public data**: WorldPop demographics, OpenStreetMap roads and rail, nighttime lights, vegetation indices, flood and drought layers

**Government sources**: Admin-level socio-economic data from RAPP and poverty mapping covering infrastructure, communications, market access, water resources, climate events, poverty levels, food security, and production constraints

All data gets aligned to the 1-km analysis grid, with vector layers rasterized or summarized consistently.

## What this project delivers

The analysis provides comparable metrics for any corridor province and its municipalities. Key features include:

- A customizable priority surface that weighs population, cropland, accessibility, and risk factors
- Pixel-level analysis converted into actionable project clusters  
- Service catchment areas calculated using road networks and travel times
- Overlay analysis with existing government, World Bank, and other investments
- Origin-destination flow modeling between municipalities

The system uses a consistent 1-km grid with rioxarray/xarray tools and careful coordinate system handling for reproducible results.

## Quick start

**Repository structure:**

```
src/          # Processing steps 00-14 plus geo utilities
notebooks/    # Pipeline runner and analysis chapters  
data/         # Input vectors and rasters by area
outputs/      # Results: rasters, tables, figures
docs/         # Jupyter book documentation
```

**Setup:**

1. Set `PROJECT_ROOT` and `AOI` environment variables
2. Edit paths and parameters in `src/config.py` 
3. Run processing scripts in sequence (00 through 14)
4. Save the provenance JSON from Chapter 14
5. Browse results in the Jupyter Book documentation

## Reproducibility & documentation

The system enforces proper georeferencing when writing rasters and maintains consistent coordinate systems throughout. Chapter 14 generates a complete provenance record with area parameters and file hashes for full reproducibility.

## License & acknowledgments

© World Bank & partners. Earth Observation and public datasets used under their respective licenses.

This tool builds on public datasets from WorldPop, OpenStreetMap, and others, plus Angola's RAPP and Poverty Map initiatives.

---

## Contact

**Geospatial Operations Support Team ([GOST](https://worldbank.github.io/GOST/README.html))**  
Development Economics Data Group, The World Bank

Benjamin Stewart, Katie Williams, Benny Istanto

---

``````{admonition} Map Disclaimer
:class: dropdown
Country borders or names do not necessarily reflect the World Bank Group's official position. This map is for illustrative purposes and does not imply the expression of any opinion on the part of the World Bank, concerning the legal status of any country or territory or concerning the delimitation of frontiers or boundaries.
``````
---

*This repository is actively maintained. We welcome feedback and contributions as we continue developing the methodology.*
# Lobito Corridor Spatial Analysis

> **Project Reference:** Diversifica Mais ([P178035](https://opswork.worldbank.org/home/P178035))

The [**Lobito Corridor**](https://www.lobitocorridor.org/history-background) connects Angola's Atlantic gateway at the **Port of Lobito** to inland markets via the historic **Benguela Railway (CFB)**. The corridor crosses **Benguela, Huambo, Bié, Moxico and Moxico-Leste** (Moxico-Leste became a new province in 2024, carved from eastern Moxico municipalities) before reaching the **Luau** border and connecting to the Copperbelt.

It's more than just a transport route—it's an investment platform where better last-mile connections, services, and market access can unlock real gains for smallholders and rural communities. This analysis turns open geospatial data into clear, prioritized recommendations along that spine.

![Lobito-Corridor](./images/AGO_A4L_LC_Admin.png)
*Figure: Lobito Corridor alignment and intersecting provinces*

---

## The questions we help answer

Our team uses this analysis to answer:

1. **Impact assessment**: How many households benefit from each project, especially the poorest? Which areas maximize cropland and access improvements?

2. **Spatial coordination**: Are projects clustered to reinforce each other or spread too thin?

3. **Equity**: Do investment priorities align with rural poverty and food insecurity patterns?

4. **Synergies**: How close are proposed projects to existing World Bank, government, and other investments? Where can we coordinate?

5. **Logistics**: Do planned upgrades improve market and finance access while supporting the Caála Logistics Platform and other key nodes?

---

## Read this first — scope & data reality

This tool focuses on [**Angola**](https://www.worldbank.org/en/country/angola), prioritizing provinces along the Lobito Corridor railway. Since official, detailed datasets are often limited or inconsistent, we use **Earth Observation** and **public geospatial data** ([WorldPop](https://hub.worldpop.org/geodata/summary?id=72366), [OpenStreetMap](https://download.geofabrik.de/africa/angola.html), [global vegetation indices](https://doi.org/10.5067/MODIS/MOD13Q1.061), [flood](https://www.fathom.global/product/global-flood-map/)/[drought](https://www.fao.org/giews/earthobservation/asis/index_1.jsp?lang=en) layers, [nighttime lights](https://eogdata.mines.edu/products/vnl/), etc.) plus government socio-economic data from [RAPP](https://andine.ine.gov.ao/nada/index.php/catalog/30) and poverty mapping.

What this means for you:

* Results are **comparable across provinces** but **approximate** at fine scales
* Travel times, accessibility, and risk layers are **modeled**—local knowledge should refine them
* We use **transparent controls** (weights, thresholds, masks) so teams can test different assumptions when data is limited

**Use this to prioritize and coordinate**, then validate with local engineering expertise, budgets, and field knowledge.

---

## What you'll get

The analysis provides comparable metrics for any corridor province and its municipalities:

* A customizable priority surface that weighs population, cropland, accessibility, and risk factors
* Pixel-level analysis converted into **actionable project clusters**
* **Service catchment** areas calculated using road networks and travel times
* **Synergy analysis** with existing government, World Bank, and other investments
* **Origin-destination flow modeling** between municipalities to understand corridor movement patterns

The system uses a consistent 1-km grid with rioxarray/xarray tools and careful coordinate system handling for reproducible results

---

## Quick tour

``````{admonition} Click the button to reveal!
:class: dropdown

1. **[If we could only start in three places…](chapters/01-decide-first.md)**
   
   A 60-second snapshot: top clusters and municipalities to target first, with key numbers (people within 60 minutes, cropland coverage, electrification gaps).

2. **[Are we putting scarce resources where they matter most?](chapters/02-muni-shortlist.md)**
   
   A ranked shortlist using a transparent scoring system, see and adjust the rules behind the rankings.

3. **[Where are the actionable hotspots?](chapters/03-actionable-hotspots.md)**
   
   Priority clusters: how big they are, who lives and farms there, and which municipalities they cover.

4. **[What happens when we change the rules?](chapters/04-scenarios.md)**
   
   Scenario testing (drop nighttime lights, raise cropland thresholds, fix area limits): what stays stable, what changes, and why.

5. **[Who benefits within 30/60/120 minutes?](chapters/05-catchments.md)**
   
   Road-aware catchments from project sites showing beneficiaries and cropland served at different travel times.

6. **[Can we coordinate with other investments?](chapters/06-synergies.md)**
   
   Proximity to government, World Bank, and other projects: distances and counts within 5/10/30 km for coordination opportunities.

7. **[Do priorities align with poverty and food insecurity?](chapters/07-equity-lens.md)**
   
   Equity check: correlations and outliers at municipality level to validate targeting approaches.

8. **[How do places in the corridor interact?](chapters/08-movement.md)**
   
   Flow patterns between municipalities and sample trips, where exchanges are strongest under current conditions.

9. **[Change the settings, not the code](chapters/09-cheatsheet.md)**
   
   Quick reference to adjust weights, masks, thresholds, and area limits before re-running the analysis.

10. **[Run this anywhere along the corridor](chapters/10-aoi-playbook.md)**
    
    Simple steps to switch to different areas or provinces and recreate the analysis.

11. **[What data goes in?](chapters/11-data-menu.md)**
    
    Complete list of input data: rasters and municipality-level themes with units and resolution.

12. **[How it works](chapters/12-how-it-works.md)**
    
    The logic behind data processing, scoring, clustering, travel time calculations, and flow modeling.

13. **[Handle with care](chapters/13-limits.md)**
    
    Known limitations (OpenStreetMap coverage, proxy indicators) and how to quality-check outputs.

14. **[Documentation & reproducibility](chapters/14-provenance.md)**
    
    Complete record of area, date, parameters, and file paths used for results.

15. **[Appendix for analysts](chapters/15-appendix.md)**
    
    Full parameter tables, equations, script descriptions, and outputs inventory.
``````

---

## Repository structure

```
src/          # Processing steps 00-14 plus geo utilities
notebooks/    # Pipeline runner and analysis chapters  
data/         # Input vectors and rasters by area
outputs/      # Results: rasters, tables, figures
docs/         # Jupyter book documentation
```

---

## Current status

* Core 1-km data processing, vector alignment, flood aggregation, and reliable output generation
* Flexible priority scoring with area/percentage selection and cluster analysis  
* Catchment analysis, synergy mapping, and origin-destination flow modeling
* Complete documentation with copy-paste code examples and parameter guides

---

## What's coming next

TBD

---

## Who this is for

* **Policy & program leads**: Decide where to invest first and coordinate efforts  
* **Infrastructure engineers**: Examine clusters and catchments for project planning  
* **Analysts**: Run scenarios and test how stable priorities are under different assumptions

---

## How to use this guide

* Start with **Chapters 1-2** for the **quick answer** and **priority list**  
* Jump to **Chapter 3** for **priority clusters** or **Chapter 5** for **service catchments**  
* Use **Chapters 4 & 9** to **adjust settings** without coding  
* Check **Chapters 13-14** for **limitations** and **complete documentation**

---

## Getting started

1. Set up your environment:

   ```python
   import os
   os.environ["PROJECT_ROOT"] = "/path/to/your/project"
   os.environ["AOI"] = "moxico"  # or huambo, benguela, moxicoleste…
   ```

2. Review `src/config.py` for data paths and parameters (weights, masks, road speeds)

3. Run processing steps **00 → 14** from `src/` or use the provided notebooks, then explore the chapters

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

*This guide is actively maintained and updated as we improve our methods. We welcome feedback and contributions.*

# All pipeline code

Below is a single-page view of the pipeline scripts in `src/`. Expand only what you need. The code is included live from the repository so it stays in sync.

````{admonition} Tip
You can use the search box on this page to find functions or parameters across steps.
````

````{dropdown} Step 00 — Align & rasterize
:open:
```{literalinclude} ../../src/step_00_align_and_rasterize.py
:language: python
:linenos:
:caption: step_00_align_and_rasterize.py
```
````

````{dropdown} Step 01 — Isochrones
```{literalinclude} ../../src/step_01_isochrones.py
:language: python
:linenos:
:caption: step_01_isochrones.py
```
````


````{dropdown} Step 02 — Iso KPIs (population, cropland, electrification)
```{literalinclude} ../../src/step_02_kpis_population_cropland_electric.py
:language: python
:linenos:
:caption: step_02_kpis_population_cropland_electric.py
```
````


````{dropdown} Step 03 — Legacy priority (optional)
```{literalinclude} ../../src/step_03_priority_surface.py
:language: python
:linenos:
:caption: step_03_priority_surface.py
```
````


````{dropdown} Step 04 — Flood bottlenecks (priority × flood)
```{literalinclude} ../../src/step_04_flood_bottlenecks_from_road_raster.py
:language: python
:linenos:
:caption: step_04_flood_bottlenecks_from_road_raster.py
```
````


````{dropdown} Step 05 — Site audit points
```{literalinclude} ../../src/step_05_site_audit_points.py
:language: python
:linenos:
:caption: step_05_site_audit_points.py
```
````


````{dropdown} Step 06 — Admin2 ingest (RAPP themes)
```{literalinclude} ../../src/step_06_muni_ingest.py
:language: python
:linenos:
:caption: step_06_muni_ingest.py
```
````


````{dropdown} Step 07 — Priority (tunable)
```{literalinclude} ../../src/step_07_priority_tunable.py
:language: python
:linenos:
:caption: step_07_priority_tunable.py
```
````


````{dropdown} Step 08 — Project KPIs (near-priority & access)
```{literalinclude} ../../src/step_08_project_kpis.py
:language: python
:linenos:
:caption: step_08_project_kpis.py
```
````


````{dropdown} Step 09 — Municipality targeting (scenario sweep)
```{literalinclude} ../../src/step_09_muni_targeting.py
:language: python
:linenos:
:caption: step_09_muni_targeting.py
```
````


````{dropdown} Step 10 — Priority scenarios (consolidation)
```{literalinclude} ../../src/step_10_priority_scenarios.py
:language: python
:linenos:
:caption: step_10_priority_scenarios.py
```
````


````{dropdown} Step 11 — Priority clusters
```{literalinclude} ../../src/step_11_priority_clusters.py
:language: python
:linenos:
:caption: step_11_priority_clusters.py
```
````


````{dropdown} Step 12 — Traveltime catchments
```{literalinclude} ../../src/step_12_traveltime_catchments.py
:language: python
:linenos:
:caption: step_12_traveltime_catchments.py
```
````


````{dropdown} Step 13 — Synergies overlay
```{literalinclude} ../../src/step_13_synergies_overlay.py
:language: python
:linenos:
:caption: step_13_synergies_overlay.py
```
````


````{dropdown} Step 14 — OD-Lite (gravity + agents)
```{literalinclude} ../../src/step_14_od_lite.py
:language: python
:linenos:
:caption: step_14_od_lite.py
```
````


````{dropdown} Shared utilities — utils_geo.py
```{literalinclude} ../../src/utils_geo.py
:language: python
:linenos:
:caption: utils_geo.py
```
````

````{dropdown} Shared utilities — config.py
```{literalinclude} ../../src/config.py
:language: python
:linenos:
:caption: config.py
```
````

# Data dictionary & sources

This page lists the core inputs used across provinces and the Admin2 socio-economic themes. It’s a single point of reference so users know what each layer means and how it’s used.

## A. Core geospatial layers (public/EO)

1. **Population (WorldPop, 1 km)**  
   - Purpose: beneficiaries, OD gravity mass, normalizing per-km² stats.  
   - Units: persons per 1-km cell.  
   - Path pattern: `data/rasters/ago_pop_{AOI}_2025_1km.tif` (aligned to grid).

2. **Travel time to market/finance (minutes)**  
   - Purpose: access, catchments (30/60/120 min), site KPIs.  
   - Units: minutes per cell (cost surface).  
   - Path: `data/rasters/ago_phy_{AOI}_traveltime_market.tif`.

3. **Night-time lights (VIIRS, 300 m resampled to 1 km)**  
   - Purpose: proxy for economic density (optional).  
   - Units: normalized 0–1.  
   - Path: `data/rasters/ago_phy_{AOI}_viirs_ntl_2024.tif`.

4. **Vegetation index (1 km)**  
   - Purpose: proxy for vegetation/greenness (optional).  
   - Units: 0–1.  
   - Path: `data/rasters/ago_phy_{AOI}_vegindex_mean_2024.tif`.

5. **Drought frequency (1 km)**  
   - Purpose: risk tempering in priority scoring.  
   - Units: % of severe drought frequency (0–100).  
   - Path: `data/rasters/ago_phy_{AOI}_asishdfc_all_al30_2024.tif`.

6. **Flood depth (30 m → aggregated to 1 km: max)**  
   - Purpose: bottleneck screening, risk in clusters.  
   - Units: meters (max depth in 1-km cell).  
   - Path: `data/rasters/ago_phy_{AOI}_pluvialdefended_100rp_2020.tif`.

7. **Cropland (vector → fraction/presence raster)**  
   - Purpose: crop potential & thresholds for selection.  
   - Outputs (derived): `{AOI}_cropland_fraction_1km.tif`, `{AOI}_cropland_presence_1km.tif`.

8. **Electrification & settlement masks (vector → raster)**  
   - Purpose: focus on unelectrified / rural if required.  
   - Outputs: `{AOI}_elec_grid_1km.tif`, `{AOI}_elec_unelectrified_1km.tif`, `{AOI}_urban_1km.tif`, `{AOI}_rural_1km.tif`.

## B. Admin2 socio-economic (RAPP & poverty map)

Files follow: `data/vectors/admin2_rapp/ago_gov_{adm2}_{theme}_rapp_2020_a.shp`  

- Key columns: `NAM_1` (Admin1), `NAM_2` (Admin2), `ADM2CD_c`, and `data1..dataN` per theme.

Themes (examples of `data#` meanings):

1. **waterresources** — rivers, streams, lakes, lagoons, boreholes.
2. **communications** — telephone, internet, newspaper, radio, TV, none.  
3. **infra** — electricity, water storage, veterinarians, banks, mech. equipment, ag schools, primary schools, field schools, health units, ag product stock.  
4. **foodinsecurity** — 8-item HFI indicators + scale.  
5. **outflow** — access/transport difficulties (insufficient means, high cost, etc.).  
6. **poverty** — rural, urban, total poverty rates.  
7. **productions** — constraints: land, water, labor, credit, equipment, etc.  
8. **traveltime** — avg hours to market/finance.  
9. **climevents** — prolonged drought, drought, strong winds, excessive rain, floods.

## C. Outputs (by AOI)

- Rasters in `outputs/rasters/{AOI}_*.tif` (1-km stack, selection masks, clusters).  
- Tables in `outputs/tables/{AOI}_*.csv` (municipality ranks, clusters KPIs, catchments, synergies, OD flows, scenarios).  
- Repro summary in `outputs/tables/{AOI}_provenance.json` (parameters & file hashes).

```{admonition} Conventions
- All rasters share the **same CRS and transform** (EPSG:4326; ~1-km resolution).  
- Vector-to-raster steps use the same target grid; flood is aggregated by **max** from 30 m.  
- File names are **AOI-prefixed** (e.g., `moxico_…`), so multiple provinces can co-exist.

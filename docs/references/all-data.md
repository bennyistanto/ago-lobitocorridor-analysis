# Data Dictionary & Sources

> **Last updated:** 2026-03-30

This document lists every input variable used in the Angola Lobito Corridor Spatial Analysis, what it measures, and how it contributes to the pipeline. It is the single point of reference for understanding what data goes in and why.

---

## A. Core Geospatial Layers

### A1. Population

| Field | Detail |
|-------|--------|
| **Variable** | Population count |
| **Source** | WorldPop 2025 (constrained, UN-adjusted) |
| **File** | `ago_pop_{AOI}_2025_CN_1km_R2025A_v1.tif` |
| **Resolution** | ~1 km |
| **Unit** | Persons per grid cell |
| **Role in pipeline** | Beneficiary counts in catchments, OD gravity model mass, per-capita normalisation, cluster KPIs. Used in every step from isochrone coverage to equity analysis. |
| **Transform** | Linear min-max scaling, capped at 95th percentile to reduce outlier dominance (e.g. Lobito city). |
| **Weight** | `W_POP` - typically 0.10–0.25 depending on preset. |

### A2. Travel Time to Market / Financial Centre

| Field | Detail |
|-------|--------|
| **Variable** | Pre-computed travel time to nearest city centre |
| **Source** | World Bank GOST (Global Operational Support Team) - internally computed using gost python package and OSM road network |
| **File** | `ago_phy_{AOI}_traveltime_market.tif` |
| **Resolution** | ~1 km |
| **Unit** | Minutes |
| **Role in pipeline** | Template grid (all other layers align to this). Defines accessibility component of priority score. Drives isochrone thresholds (30/60/120/240 min). Used for catchment delineation. |
| **Transform** | Sigmoid (inflection at 60–120 min depending on preset), **inverted** so shorter travel time → higher accessibility score. |
| **Weight** | `W_ACC` - typically 0.10–0.35, highest in connectivity preset. |
| **Note** | This is NOT a friction surface. It is a pre-computed travel time raster (minutes to nearest city). The friction surfaces (Section A8) are separate inputs used for facility access computation. |

### A3. Night-Time Lights (VIIRS) - 3 Variables

The pipeline uses **three** NTL-derived layers, not one. Each captures a different aspect of economic activity:

#### A3a. NTL 10-Year Mean Radiance

| Field | Detail |
|-------|--------|
| **Variable** | Mean annual night-time light radiance (2015–2024) |
| **Source** | NOAA VIIRS Day/Night Band, annual composites |
| **File** | `ago_phy_{AOI}_ntl_mean_2015_2024.tif` |
| **Resolution** | ~464 m → resampled to 1 km |
| **Unit** | nW cm⁻² sr⁻¹ (nanowatts per square centimetre per steradian) |
| **What it measures** | Average brightness level over 10 years. Higher values = more persistent economic activity (electrified settlements, industry, commerce). |
| **Role in pipeline** | Economic activity proxy in priority scoring. Brighter areas indicate existing infrastructure and market presence. |
| **Transform** | Logarithmic compression (`log(x + 0.01)`, capped at 95th percentile). This compresses the huge range between dark rural areas and bright cities. |
| **Weight** | `W_NTL` - 0.00 in balanced/food_security, 0.20–0.25 in connectivity/peri-urban presets. |
| **Presets using it** | Connectivity, Peri-Urban Growth. Turned off in food security and climate resilience (where lights are less relevant). |

#### A3b. NTL Trend - Absolute Slope

| Field | Detail |
|-------|--------|
| **Variable** | Linear trend in annual NTL radiance (2015–2024) |
| **Source** | Derived from NOAA VIIRS annual composites |
| **File** | `ago_phy_{AOI}_ntl_trend_slope_2015_2024.tif` |
| **Resolution** | ~464 m → resampled to 1 km |
| **Unit** | Radiance per year (nW cm⁻² sr⁻¹ yr⁻¹) |
| **What it measures** | Whether a location is getting brighter (economic growth) or dimmer (decline) over the decade. Positive = growing electrification/activity. |
| **Role in pipeline** | KPI reporting; identifies areas of economic expansion vs. stagnation. Available for overlay analysis but not directly weighted in current presets. |

#### A3c. NTL Trend - Relative (% per Year)

| Field | Detail |
|-------|--------|
| **Variable** | Relative annual change in NTL radiance |
| **Source** | Derived from NOAA VIIRS annual composites |
| **File** | `ago_phy_{AOI}_ntl_trend_pctyr_2015_2024.tif` |
| **Resolution** | ~464 m → resampled to 1 km |
| **Unit** | Percent per year (%/yr) |
| **What it measures** | Rate of brightness change normalised by baseline level. A dim area that doubles is +100%/yr; a bright city adding the same absolute radiance may only be +2%/yr. Captures emerging economic centres. |
| **Role in pipeline** | KPI and diagnostic. Useful for identifying "rising" areas that may not yet appear bright in the mean layer. |

> **Legacy fallback:** If multi-year NTL is unavailable, the pipeline falls back to `ago_phy_{AOI}_viirs_ntl_2024.tif` (single-year 2024 snapshot, ~300 m).

### A4. Vegetation & Agricultural Productivity (MCD12Q2) - 6 Variables

These replaced the single NDVI layer from v1. They come from MODIS Land Cover Dynamics (MCD12Q2), which tracks crop phenology:

#### A4a. Growing Season Length (GSL) - Median

| Field | Detail |
|-------|--------|
| **Variable** | Median growing season length across 24 years |
| **Source** | MODIS MCD12Q2 v061, 2001–2024 |
| **File** | `ago_phy_{AOI}_gsl_median_days_2001_2024.tif` |
| **Resolution** | 500 m → resampled to 1 km |
| **Unit** | Days |
| **What it measures** | How long the growing season typically lasts. 120–270 days is the productive agricultural window; <90 days is too short for most crops; >300 days indicates perennial forest (not cropland). |
| **Role in pipeline** | Primary vegetation component (replaces NDVI). Used in food security and climate resilience presets. |
| **Transform** | Trapezoidal (0 below 90 days → ramps to 1 at 120 → plateau to 270 → ramps to 0 by 300). Targets the "Goldilocks" agricultural window. |
| **Weight** | `W_VEG` - 0.20–0.25 in food_security/climate_resilience; 0.00 in connectivity. |

#### A4b. GSL Trend

| Field | Detail |
|-------|--------|
| **Variable** | Linear trend in growing season length |
| **Source** | MODIS MCD12Q2, 2001–2024 |
| **File** | `ago_phy_{AOI}_gsl_trend_daysyr_2001_2024.tif` |
| **Unit** | Days per year |
| **What it measures** | Whether the growing season is getting longer (positive) or shorter (negative). Shortening seasons signal climate stress on agriculture. |
| **Role in pipeline** | KPI in climate resilience preset; diagnostic for agricultural planning. |

#### A4c. Greenup Onset Variability

| Field | Detail |
|-------|--------|
| **Variable** | Standard deviation of greenup (start-of-season) date |
| **Source** | MODIS MCD12Q2, 2001–2024 |
| **File** | `ago_phy_{AOI}_greenup_stdev_days_2001_2024.tif` |
| **Unit** | Days (standard deviation) |
| **What it measures** | How unpredictable the start of the growing season is. High variability = farmers cannot reliably time planting, increasing crop failure risk. |
| **Role in pipeline** | KPI in climate resilience and food security presets. Areas with high variability need climate-smart agricultural support. |

#### A4d. EVI Integrated Area (Seasonal Productivity)

| Field | Detail |
|-------|--------|
| **Variable** | Median integrated EVI over the growing season |
| **Source** | MODIS MCD12Q2, 2001–2024 |
| **File** | `ago_phy_{AOI}_evi_area_median_2001_2024.tif` |
| **Unit** | Index × days (area under the EVI curve) |
| **What it measures** | Total photosynthetic production during the growing season. Higher = more productive vegetation. Combines season length with peak greenness. |
| **Role in pipeline** | Food security KPI; differentiates high-yield from marginal cropland. |

#### A4e. EVI Amplitude

| Field | Detail |
|-------|--------|
| **Variable** | Median peak-to-trough EVI difference |
| **Source** | MODIS MCD12Q2, 2001–2024 |
| **File** | `ago_phy_{AOI}_evi_amplitude_median_2001_2024.tif` |
| **Unit** | Index (0–1 scale) |
| **What it measures** | Strength of the seasonal vegetation cycle. High amplitude = strong crop signal; low = evergreen forest or barren land. |
| **Role in pipeline** | Diagnostic KPI; helps distinguish active cropland from forest or grassland. |

#### A4f. Number of Cropping Cycles

| Field | Detail |
|-------|--------|
| **Variable** | Modal number of cropping cycles per year |
| **Source** | MODIS MCD12Q2, 2001–2024 |
| **File** | `ago_phy_{AOI}_numcycles_mode_2001_2024.tif` |
| **Unit** | Count (1, 2, or rarely 3) |
| **What it measures** | How many harvests occur per year. Double-cropping areas have higher food production potential. |
| **Role in pipeline** | Food security KPI; identifies areas with multi-crop potential. |

> **Legacy fallback:** If MCD12Q2 layers are unavailable, falls back to `ago_phy_{AOI}_vegindex_mean_2024.tif` (single-year NDVI, 0–1).

### A5. Drought & Climate Stress (SPEI-12) - 9 Variables

These replaced the single FAO ASI drought layer from v1. SPEI-12 is a 12-month Standardised Precipitation–Evapotranspiration Index computed using run theory over 65 years:

#### A5a. Number of Drought Events (Full Period)

| Field | Detail |
|-------|--------|
| **Variable** | Count of drought events (SPEI-12 < −1.0 threshold) |
| **Source** | SPEI Global Drought Monitor v3, 1958–2025 |
| **File** | `ago_cli_{AOI}_spei12_num_events_1958_2025.tif` |
| **Resolution** | ~4 km → resampled to 1 km |
| **Unit** | Count |
| **What it measures** | How many distinct drought episodes a location has experienced over 65 years. More events = chronically drought-prone area. |
| **Role in pipeline** | Primary drought component in priority scoring. Areas with more events get higher drought-risk scores. |
| **Transform** | Sigmoid (inflection at 6 events) or threshold (ramp from 3 to 10 events), depending on preset. |
| **Weight** | `W_DRT` - 0.10 in balanced, **0.30** in climate_resilience. |

#### A5b. Maximum Drought Duration

| Field | Detail |
|-------|--------|
| **Variable** | Longest single drought episode |
| **File** | `ago_cli_{AOI}_spei12_max_duration_1958_2025.tif` |
| **Unit** | Months |
| **What it measures** | Duration of the worst drought on record. Long droughts (>12 months) devastate perennial crops and livestock. |
| **Role in pipeline** | Climate resilience KPI. |

#### A5c. Mean Drought Intensity

| Field | Detail |
|-------|--------|
| **Variable** | Average severity of drought events |
| **File** | `ago_cli_{AOI}_spei12_mean_intensity_1958_2025.tif` |
| **Unit** | SPEI index (negative; more negative = more severe) |
| **What it measures** | How harsh droughts typically are. An area may have few events but each is devastating. |
| **Role in pipeline** | Climate resilience KPI. |

#### A5d. Minimum Peak (Worst Month)

| Field | Detail |
|-------|--------|
| **Variable** | Most extreme single-month SPEI value ever recorded |
| **File** | `ago_cli_{AOI}_spei12_min_peak_1958_2025.tif` |
| **Unit** | SPEI index (most negative) |
| **What it measures** | The absolute worst drought month in 65 years. Useful for extreme-event planning. |
| **Role in pipeline** | Climate resilience diagnostic. |

#### A5e. Mean Drought Magnitude

| Field | Detail |
|-------|--------|
| **Variable** | Average cumulative departure during drought events |
| **File** | `ago_cli_{AOI}_spei12_mean_magnitude_1958_2025.tif` |
| **Unit** | Cumulative SPEI-months |
| **What it measures** | Combines duration × intensity. High magnitude = prolonged severe droughts. |
| **Role in pipeline** | Climate resilience KPI. |

#### A5f–g. Recent Period Variants (2001–2025)

| Variable | File |
|----------|------|
| Num events (recent) | `ago_cli_{AOI}_spei12_num_events_2001_2025.tif` |
| Max duration (recent) | `ago_cli_{AOI}_spei12_max_duration_2001_2025.tif` |

These allow comparison of recent drought exposure vs. the full 65-year record to detect worsening trends.

#### A5h. Baseline Period (1961–1990)

| Variable | File |
|----------|------|
| Num events (baseline) | `ago_cli_{AOI}_spei12_num_events_1961_1990.tif` |

Reference period for climate-normals comparison (baseline shift analysis).

#### A5i. SPI-3 Short-Term Agricultural Drought

| Field | Detail |
|-------|--------|
| **Variable** | Short-term (3-month) precipitation deficit events |
| **File** | `ago_cli_{AOI}_spi03_num_events_1958_2025.tif` |
| **Unit** | Count |
| **What it measures** | Quick-onset dry spells that damage annual crops within a single season. Complements SPEI-12 which captures multi-season drought. |
| **Role in pipeline** | Food security diagnostic KPI. |

> **Legacy fallback:** `ago_phy_{AOI}_asishdfc_all_al30_2024.tif` (FAO ASI 2024, % severe drought).

### A6. Flood Risk

| Field | Detail |
|-------|--------|
| **Variable** | Pluvial flood maximum depth (100-year return period) |
| **Source** | Fathom / JBA, 2020 |
| **File** | `ago_nhr_{AOI}_pluvialdefended_100rp_2020.tif` |
| **Resolution** | 30 m → aggregated to 1 km (max depth per cell) |
| **Unit** | Metres |
| **Role in pipeline** | Road-flood bottleneck identification (Step 04): cells with depth ≥0.3 m on road pixels are flagged as flood-risk. Used in cluster scoring (cropland-flood overlap) and intervention simulation (what-if road upgrades). |

### A7. Cropland

| Field | Detail |
|-------|--------|
| **Variable** | Cropland fraction per 1-km cell |
| **Source** | ESA WorldCover 2021 (Class 40 - Cropland) |
| **File** | `ago_phy_{AOI}_cropland_100m_worldcover.tif` |
| **Resolution** | 100 m → zonal fraction at 1 km |
| **Unit** | 0–1 fraction (e.g. 0.45 = 45% of cell is cropland) |
| **Role in pipeline** | Agricultural potential indicator. Used as hard filter in food security preset (cells with <20% cropland excluded). Cropland area within catchments is a key KPI. |
| **Hard filter** | `MASK_MIN_CROPLAND` - 0.05 (balanced), 0.10 (peri-urban), 0.20 (food security). |

> **Legacy fallback:** `ago_phy_{AOI}_cropland_10m_worldcover_a.shp` (vector polygons, rasterised if raster unavailable).

### A8. Friction Surfaces & Facility Travel Times - 4 Variables

These are **separate from** the travel-time-to-market raster (A2). They are cost surfaces used to compute travel time to the nearest health or education facility:

#### A8a–b. MAP Friction Surfaces

| Variable | File | Unit | What it measures |
|----------|------|------|-----------------|
| Motorised friction | `ago_phy_{AOI}_motorized_frictionsurface_map_2019.tif` | min/metre | Time cost to traverse each cell by vehicle. Accounts for road type, terrain, land cover. |
| Walking friction | `ago_phy_{AOI}_walking_frictionsurface_map_2019.tif` | min/metre | Time cost on foot. Used for areas without road access. |

#### A8c–d. Derived: Travel Time to Nearest Facility

| Variable | Output file | Unit | Method |
|----------|-------------|------|--------|
| TT to health (motorised) | `{AOI}_tt_health_motorised_1km.tif` | Minutes | Minimum-cost-path from every cell to nearest OSM health facility, using motorised friction |
| TT to education (motorised) | `{AOI}_tt_education_motorised_1km.tif` | Minutes | Same method, to nearest OSM school/university |

| Field | Detail |
|-------|--------|
| **Facility sources** | `ago_poi_{AOI}_health_facilities_p.shp` (OSM), `ago_poi_{AOI}_education_facilities_p.shp` (OSM) |
| **Role in pipeline** | Last-mile access preset weights these heavily (`W_TT_HEALTH=0.20`, `W_TT_EDUCATION=0.15`). Areas far from facilities score higher (more underserved → higher priority). |
| **Transform** | Sigmoid (inflection at 60 min for health, 45 min for education), **inverted**. WHO recommends health facilities within 1 hour. |

### A9. Buildings & Settlement Density

| Field | Detail |
|-------|--------|
| **Variable** | Building footprint count and density per 1-km cell |
| **Source** | Google Open Buildings v3 (2022) |
| **File** | `ago_phy_{AOI}_openbuildings_a.shp` |
| **Derived outputs** | `{AOI}_building_count_1km.tif` (count), `{AOI}_building_density_1km.tif` (fraction of cell covered) |
| **Unit** | Count (buildings per cell) or fraction (0–1) |
| **What it measures** | Settlement consolidation and market presence. More buildings = more economic activity to leverage for project impact. |
| **Role in pipeline** | Peri-urban growth preset uses building density as a weighting factor (`W_BUILDING_DENSITY=0.20`). Also a KPI in catchment analysis (how many structures are within each site's service area). |

### A10. Energy Demand (DRE Atlas)

| Field | Detail |
|-------|--------|
| **Variable** | Unelectrified population and decentralised renewable energy demand |
| **Source** | DRE Atlas settlements database |
| **File** | `ago_phy_{AOI}_dre_settlements_a.shp` |
| **Derived outputs** | `{AOI}_dre_pop_unserved_1km.tif` (unserved population), `{AOI}_dre_demand_density_1km.tif` (energy demand) |
| **Unit** | Persons (unserved) or kWh/km² (demand density) |
| **What it measures** | Where unelectrified communities exist and how much energy demand could be met by off-grid renewable solutions. |
| **Role in pipeline** | Last-mile access preset weights DRE demand (`W_DRE_DEMAND=0.10`). Identifies communities where renewable energy investment could catalyse agricultural transformation. Also a catchment KPI. |

### A11. Relative Wealth Index

| Field | Detail |
|-------|--------|
| **Variable** | Relative Wealth Index |
| **Source** | Meta (Facebook) Data for Good, 2022 |
| **File** | `ago_pop_{AOI}_rwi_meta_2022.tif` |
| **Resolution** | ~2.4 km → resampled to 1 km |
| **Unit** | Index (−2 to +2; 0 = national average) |
| **What it measures** | Relative household wealth estimated from satellite imagery and connectivity data. Negative = poorer than average. |
| **Role in pipeline** | Equity overlay - **inverted** so poorer areas score higher (pro-poor targeting). Used in benefit incidence analysis and OD gravity model (RWI mass tilt). |
| **Weight** | `W_RWI` - typically 0.10–0.15. |

### A12. Electrification & Settlement Type

| Variable | File | What it measures |
|----------|------|-----------------|
| Electricity type | `ago_pop_{AOI}_electricity_type_a.shp` | Grid-electrified vs. off-grid settlements. `FinalElecCode2020==1` = grid; others = unelectrified. |
| Settlement type | `ago_pop_{AOI}_settlement_type_a.shp` | Urban vs. rural classification. |

**Derived rasters:** `{AOI}_elec_grid_1km.tif`, `{AOI}_rural_1km.tif`, `{AOI}_urban_1km.tif`

**Role:** Rural mask (hard filter in some presets: `MASK_REQUIRE_RURAL=True`). Electrification rate as a KPI in isochrone and catchment tables.

---

## B. Admin-2 Socio-Economic Survey Data (RAPP 2020)

Source: Government of Angola, Recenseamento Agro-Pecuário e Pescas (RAPP), 2020.
Files: `data/vectors/ago_gov_{AOI}_adm2_{theme}_rapp_2020_a.shp`

These are **municipality-level** (Admin-2) survey data, rasterised to the 1-km grid for integration with raster layers. All percentage values are normalised from 0–100 to 0–1 in the pipeline.

### B1. Poverty

| Variable | Unit | What it measures |
|----------|------|-----------------|
| `poverty_rural` | 0–1 | Rural poverty headcount ratio |
| `poverty_urban` | 0–1 | Urban poverty headcount ratio |
| `poverty_total` | 0–1 | Overall poverty headcount ratio |

**Role:** `W_POV` overlay (0.05–0.20). Higher poverty → higher priority score (pro-poor targeting).

### B2. Food Insecurity

| Variable | Unit | What it measures |
|----------|------|-----------------|
| `went_without_food` | 0–1 | Households that went entirely without food |
| `unable_eat_healthy` | 0–1 | Cannot afford a healthy diet |
| `few_types_of_food` | 0–1 | Limited dietary diversity |
| `skipped_meal` | 0–1 | Adults skipping meals |
| `ate_less_than_needed` | 0–1 | Ate less than needed |
| `ran_out_of_food` | 0–1 | Household food supply exhausted |
| `hungry_did_not_eat` | 0–1 | Went hungry without eating |
| `without_food_all_day` | 0–1 | Full day without any food |
| `food_insec_scale` | 0–1 | **Composite food insecurity index** (used in scoring) |

**Role:** `W_FOOD` overlay (0.05–0.25). Food security preset gives this the highest overlay weight.

### B3. Infrastructure Access

| Variable | What it measures |
|----------|-----------------|
| `electricity` | Electricity access rate |
| `water_storage` | Water storage facilities |
| `veterinarians` | Veterinary services availability |
| `banking` | Banking/financial services |
| `mech_agri_equip` | Mechanical agricultural equipment |
| `agri_schools` | Agricultural training schools |
| `primary_schools` | Primary school access |
| `field_schools` | Extension/field schools |
| `health_units` | Health facility access |
| `agri_stock` | Agricultural input stock |

**Role:** Diagnostic KPIs; contribute to municipality profiling.

### B4. Travel Time to Market

| Variable | Unit | What it measures |
|----------|------|-----------------|
| `avg_hours_to_market_financial` | Hours → converted to **minutes** in pipeline | Self-reported average travel time to nearest market or financial centre |

**Role:** `W_MTT` overlay (0.05–0.15). **Inverted** - longer travel → higher priority. Validates the raster-based accessibility layer.

### B5. Communications Access

| Variable | What it measures |
|----------|-----------------|
| `telephone` | Telephone access |
| `internet` | Internet access |
| `radio` | Radio access |
| `television` | Television access |
| `newspaper` | Newspaper access |
| `none` | No communication access at all |

### B6. Production Constraints

| Variable | What it measures |
|----------|-----------------|
| `diff_access_land` | Difficulty accessing land |
| `unavailable_agri_land` | Agricultural land unavailable |
| `diff_access_water` | Difficulty accessing water |
| `rural_exodus` | Rural-to-urban migration pressure |
| `diff_dispose_products` | Difficulty selling/distributing products |
| `lack_rain` | Inadequate rainfall |
| `lack_agri_equipment` | Lack of farming equipment |
| `lack_tech_assist` | Lack of technical assistance |
| `lack_manpower` | Labour shortage |
| `diff_access_credit` | Difficulty accessing credit |

### B7. Outflow / Market Access Constraints

| Variable | What it measures |
|----------|-----------------|
| `difficult_access_to_village` | Physical access barriers to settlements |
| `insufficient_transport` | Insufficient transport capacity |
| `lack_conservation_infra` | Lack of storage/cold chain infrastructure |
| `high_transport_cost` | High transport costs |
| `lack_transport_means` | No transport available |

### B8. Water Resources

| Variable | What it measures |
|----------|-----------------|
| `rivers` | River access |
| `streams` | Stream access |
| `lakes` | Lake access |
| `lagoons` | Lagoon access |
| `wells` | Well/borehole access |

### B9. Climate Events (Survey-Reported)

| Variable | What it measures |
|----------|-----------------|
| `prolonged_drought` | Reported prolonged drought occurrence |
| `drought` | Standard drought occurrence |
| `strong_winds` | Strong wind events |
| `excessive_rainfall` | Excessive rainfall events |
| `floods` | Flood occurrence |

**Note:** These are community-reported, complementing the satellite-derived SPEI drought indicators (Section A5).

---

## C. Reference & Administrative Layers

| Layer | File | Source | Purpose |
|-------|------|--------|---------|
| Province boundary (ADM1) | `ago_bnd_{AOI}_adm1_a.shp` | Government | Clipping, provincial statistics |
| Municipality boundary (ADM2) | `ago_bnd_{AOI}_adm2_a.shp` | Government | Zonal statistics, municipality ranking |
| Roads | `ago_trs_{AOI}_roads_osm_l.shp` | OpenStreetMap | Flood-road intersection, OD routing |
| Railways | `ago_trs_{AOI}_railways_osm_l.shp` | OpenStreetMap | Corridor infrastructure context |
| Project sites | `ago_poi_{AOI}_projectloc_dm_p.shp` | World Bank / GoA | Catchment origins, synergy analysis |
| Health facilities | `ago_poi_{AOI}_health_facilities_p.shp` | OSM | Cost-distance targets for TT to health |
| Education facilities | `ago_poi_{AOI}_education_facilities_p.shp` | OSM | Cost-distance targets for TT to education |

---

## D. How Variables Contribute to Analysis Steps

### D1. Priority Score (Step 07) - "Where should investment go?"

The priority score combines multiple layers into a single 0–1 surface. Each variable is first transformed to a 0–1 fuzzy membership, then combined via weighted sum (additive) or weighted geometric mean (non-compensatory):

| Component | Input Layer(s) | What higher score means | Typical weight range |
|-----------|---------------|------------------------|---------------------|
| **ACC** (Accessibility) | Travel time to market (A2) | Closer to markets / better connected | 0.10–0.35 |
| **POP** (Population) | WorldPop (A1) | More beneficiaries per cell | 0.10–0.25 |
| **VEG** (Vegetation) | GSL median (A4a) | Productive agricultural zone | 0.00–0.25 |
| **NTL** (Night Lights) | NTL mean (A3a) | Existing economic activity | 0.00–0.25 |
| **DRT** (Drought) | SPEI-12 events (A5a) | Higher climate vulnerability | 0.00–0.30 |
| **POV** (Poverty) | RAPP poverty_rural (B1) | Higher poverty incidence | 0.05–0.20 |
| **FOOD** (Food insecurity) | RAPP food_insec_scale (B2) | More food-insecure | 0.05–0.25 |
| **MTT** (Market travel) | RAPP avg travel time (B4) | Farther from markets | 0.05–0.15 |
| **RWI** (Wealth) | Meta RWI (A11) | Poorer households | 0.10–0.15 |
| **TT_HEALTH** | Travel time to health (A8c) | Farther from health facilities | 0.00–0.20 |
| **TT_EDU** | Travel time to education (A8d) | Farther from schools | 0.00–0.15 |
| **BLDG_DENS** | Building density (A9) | More settlement consolidation | 0.00–0.20 |
| **DRE** | Energy demand (A10) | Higher unmet energy demand | 0.00–0.10 |

### D2. Hard Filters (Exclusion Criteria)

Before scoring, some cells are excluded entirely:

| Filter | Rule | Presets using it |
|--------|------|-----------------|
| Rural only | Exclude urban cells | Balanced, Last-Mile, Food Security, Climate |
| Minimum cropland | Exclude cells below threshold (5–20%) | All except Connectivity |
| Minimum population | Exclude cells below threshold (30–100 persons) | All presets |
| Minimum GSL | Exclude cells with <90-day growing season | Food Security |
| Minimum drought events | Exclude cells with <2 events | Climate Resilience |

### D3. Catchment & Coverage Analysis (Steps 12–13)

The following variables are aggregated within each project site's catchment area:

| KPI | Source Layer | What it tells you |
|-----|-------------|-------------------|
| Population covered | A1 | How many people the site serves |
| Cropland area | A7 | Agricultural land within reach |
| Health access rate | A8c | % of catchment pop within 60 min of health facility |
| Building count | A9 | Settlement density within catchment |
| DRE demand | A10 | Energy needs within catchment |
| Drought exposure | A5a | Climate vulnerability of catchment population |
| GSL | A4a | Agricultural productivity potential |
| RWI | A11 | Wealth profile of beneficiaries |

---

## E. Preset Profiles - Which Variables Matter Most

| Preset | Primary focus | Key variables (highest weights) | Variables excluded |
|--------|--------------|--------------------------------|-------------------|
| **Balanced** | General-purpose | Accessibility, Population, Poverty, RWI | NTL trend, Buildings, DRE, TT Health |
| **Last-Mile Access** | Remote underserved | TT Health, TT Education, DRE, Poverty | NTL, VEG |
| **Peri-Urban Growth** | Near-city efficiency | NTL, Building density, Population | VEG, Drought |
| **Food Security** | Agricultural zones | Food insecurity, GSL, Cropland, Drought | NTL |
| **Climate Resilience** | Drought/flood vulnerable | Drought (0.30), GSL, Food insecurity | NTL |
| **Connectivity** | Economic corridors | Accessibility, NTL, Market travel time, Buildings | VEG, Drought |

---

## F. Output Tables

### Rasters (`outputs/rasters/`)

| File pattern | Step | Description |
|---|---|---|
| `{AOI}_pop_1km.tif` | 00 | Population (persons/cell) |
| `{AOI}_ntl_mean_1km.tif` | 00 | NTL 10-year mean radiance |
| `{AOI}_ntl_trend_slope_1km.tif` | 00 | NTL absolute trend |
| `{AOI}_ntl_trend_pctyr_1km.tif` | 00 | NTL relative trend (%/yr) |
| `{AOI}_gsl_median_1km.tif` | 00 | Growing season length (days) |
| `{AOI}_gsl_trend_1km.tif` | 00 | GSL trend (days/yr) |
| `{AOI}_greenup_stdev_1km.tif` | 00 | Greenup onset variability |
| `{AOI}_evi_area_median_1km.tif` | 00 | EVI integrated area |
| `{AOI}_evi_amplitude_median_1km.tif` | 00 | EVI amplitude |
| `{AOI}_numcycles_mode_1km.tif` | 00 | Cropping cycles (count) |
| `{AOI}_spei12_num_events_1km.tif` | 00 | Drought event count (65-year) |
| `{AOI}_spei12_max_duration_1km.tif` | 00 | Longest drought (months) |
| `{AOI}_spei12_mean_intensity_1km.tif` | 00 | Mean drought severity |
| `{AOI}_spei12_min_peak_1km.tif` | 00 | Worst single month |
| `{AOI}_spei12_mean_magnitude_1km.tif` | 00 | Cumulative drought magnitude |
| `{AOI}_spei12_num_events_recent_1km.tif` | 00 | Recent drought events (2001–2025) |
| `{AOI}_spei12_num_events_baseline_1km.tif` | 00 | Baseline drought events (1961–1990) |
| `{AOI}_spi03_num_events_1km.tif` | 00 | Short-term drought events |
| `{AOI}_cropland_fraction_1km.tif` | 00 | Cropland fraction (0–1) |
| `{AOI}_flood_rp100_maxdepth_1km.tif` | 00 | Flood depth (RP100, max) |
| `{AOI}_tt_health_motorised_1km.tif` | 00 | Travel time to health (min) |
| `{AOI}_tt_education_motorised_1km.tif` | 00 | Travel time to education (min) |
| `{AOI}_building_count_1km.tif` | 00 | Building count per cell |
| `{AOI}_building_density_1km.tif` | 00 | Building density (fraction) |
| `{AOI}_dre_pop_unserved_1km.tif` | 00 | DRE unserved population |
| `{AOI}_dre_demand_density_1km.tif` | 00 | DRE energy demand |
| `{AOI}_elec_grid_1km.tif` | 00 | Electrification grid mask |
| `{AOI}_rural_1km.tif` / `_urban_1km.tif` | 00 | Settlement type masks |
| `{AOI}_rwi_meta_1km.tif` | 00 | Relative Wealth Index |
| `{AOI}_admin2_id_1km.tif` | 00 | Municipality label grid |
| `{AOI}_priority_score_0_1.tif` | 07 | Composite priority surface (0–1) |
| `{AOI}_priority_top10_mask.tif` | 11 | Binary selection mask |
| `{AOI}_priority_clusters_1km.tif` | 11 | Labelled cluster raster |
| `{AOI}_od_bottleneck_risk.tif` | 14 | OD flow on flood-risk roads |
| `{AOI}_sim_travel_after.tif` | 16 | Post-intervention travel surface |
| `{AOI}_sim_travel_delta.tif` | 16 | Minutes saved (positive = improvement) |

### Tables (`outputs/tables/`)

| File pattern | Step | Description |
|---|---|---|
| `{AOI}_kpis_isochrones.csv` | 02 | Province-wide isochrone statistics (pop, cropland, health access by threshold) |
| `{AOI}_municipality_indicators.csv` | 06 | Full RAPP survey data per municipality |
| `{AOI}_municipality_profiles.csv` | 06 | Key indicator subset + poverty quintile |
| `{AOI}_corr_with_rural_poverty.csv` | 06 | Correlation of each RAPP indicator with rural poverty |
| `{AOI}_priority_admin2_rank.csv` | 07 | Municipality priority ranking |
| `{AOI}_priority_muni_rank.csv` | 09 | Municipality composite score + 12 indicators |
| `{AOI}_priority_scenarios_summary.csv` | 10 | 6-preset scenario comparison |
| `{AOI}_priority_clusters.csv` | 11 | Cluster KPIs (area, pop, cropland, flood, climate, buildings, DRE) |
| `{AOI}_project_kpis.csv` | 08 | Per-site KPIs at 5/10/30 km buffers |
| `{AOI}_catchments_kpis.csv` | 12 | Catchment stats at 30/60/120/240 min |
| `{AOI}_marginal_catchment.csv` | 12 | Net-new beneficiaries per site |
| `{AOI}_site_synergies.csv` | 13 | Site proximity to existing projects |
| `{AOI}_cluster_synergies.csv` | 13 | Cluster proximity to existing projects |
| `{AOI}_od_gravity.csv` | 14 | OD flow matrix (gravity model) |
| `{AOI}_od_zone_attrs.csv` | 14 | Zone attributes (pop, coords, RWI) |
| `{AOI}_od_agents.csv` | 14 | Sampled agent trips |
| `{AOI}_od_bottleneck_cells.csv` | 14 | Flood-risk road cells ranked by flow load |
| `{AOI}_benefit_incidence.csv` | 09 | Benefit incidence curve data |
| `{AOI}_equity_summary.csv` | 09 | Concentration index and interpretation |
| `{AOI}_roads_flood_risk_summary.csv` | 04 | Road-flood risk aggregate statistics |
| `{AOI}_sim_impact_summary.csv` | 16 | Population gaining access by improvement threshold |
| `corridor_dashboard.csv` | 15 | Cross-province headline metrics |
| `corridor_cluster_inventory.csv` | 15 | All clusters across all provinces |

---

## G. Conventions

- All rasters share the **same CRS and transform** (EPSG:4326; ~1 km resolution).
- Vector-to-raster steps use the template grid; flood is aggregated by **max** from 30 m.
- File names are **AOI-prefixed** (e.g. `benguela_…`). Multiple provinces co-exist in the same output folder.
- Corridor-wide tables (Step 15) have no AOI prefix - they aggregate all provinces.
- AOI codes: `benguela`, `huambo`, `bie`, `moxico`, `moxicoleste`.
- Provenance: `outputs/tables/{AOI}_provenance.json` records parameters and file hashes for reproducibility.

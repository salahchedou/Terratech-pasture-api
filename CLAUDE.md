# CLAUDE.md

Guidance for Claude when working in this repository.

## What this project is

A predictive pipeline and mobile app backend that helps cattle herders in
Mauritania find nutritious pasture. Satellite imagery is turned into pasture
quality maps, and the results are sent to herders in a very lightweight form.

## Data

- Source: Sentinel-2 Level-2A (surface reflectance), bands B02, B03, B04, B08,
  B8A, B11, B12 plus the SCL scene classification layer for cloud/shadow masking.
- Area of interest: Mauritania, focus on the Sahelian pastoral belt in the south
  (Hodh Ech Chargui, Hodh El Gharbi, Assaba, Gorgol, Brakna, Trarza, Guidimakha).
- Seasonality matters: the rainy season is roughly July to October, the dry
  season pasture decline runs November to June.

## Domain rules (do not break these)

- Soils are sandy and highly reflective. Use **SAVI** (L = 0.5) rather than plain
  NDVI as the main vegetation index. NDVI may be kept only as a comparison feature.
- Use moisture indices too, mainly **NDMI** ((B8A - B11) / (B8A + B11)).
- Always mask clouds, cloud shadows and saturated pixels with SCL before
  computing indices.
- Outputs must be tiny. Herders have very low bandwidth. Prefer:
  - vector data (simplified GeoJSON polygons or points, coordinates rounded to
    ~4 decimals), and
  - short SMS/USSD alert text (160 characters max, plain ASCII, French and
    Hassaniya Arabic transliteration).
  Never ship raster tiles to the mobile client.

## Modelling

- Task: regression of pasture quality (biomass or a nutrition score) from
  spectral indices, plus terrain and rainfall features where available.
- Validate with spatial or temporal splits, never a random pixel split, to avoid
  leakage between neighbouring pixels.
- Report RMSE, MAE and R² per region and per season.

## Stack and conventions

- Python 3.11+. Suggested libraries: rasterio, xarray / rioxarray, geopandas,
  shapely, scikit-learn or LightGBM, pystac-client for imagery search.
- Suggested layout:
  - `pipeline/` data download, masking, index computation
  - `model/` training, evaluation, inference
  - `api/` backend that serves vector results and builds SMS/USSD messages
  - `tests/` pytest tests
  - `notebooks/` exploration only, no production logic
- Keep large data out of git (`data/` is git-ignored).
- Run tests with `pytest` before committing.

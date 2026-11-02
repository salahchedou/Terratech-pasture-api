# CLAUDE.md

Guidance for Claude when working in this repository.

## What this project is

A dual-system pipeline and mobile app backend that helps cattle herders in Mauritania find nutritious pasture. 
The architecture is strictly divided into two distinct components:
1. **5-Day Surveillance Tracker:** A near real-time operational monitor that pulls fresh imagery every 5 days for specific grids (e.g., a 50x50km bounding box) to report current pasture health.
2. **Seasonal AI Predictor:** A machine learning forecasting model that predicts dry-season biomass and crude protein drop-off based on historical data and rainy season accumulation.

Outputs from both systems are sent to herders in a very lightweight form.

## Data

- Source: Sentinel-2 Level-2A (surface reflectance), bands B02, B03, B04, B08, B8A, B11, B12 plus the SCL scene classification layer for cloud/shadow masking.
- Area of interest: Mauritania, focus on the Sahelian pastoral belt in the south (Hodh Ech Chargui, Hodh El Gharbi, Assaba, Gorgol, Brakna, Trarza, Guidimakha).
- Seasonality matters: the rainy season is roughly July to October, the dry season pasture decline runs November to June.

## Domain rules (do not break these)

- Soils are sandy and highly reflective. Use **SAVI** (L = 0.5) rather than plain NDVI as the main vegetation index. NDVI may be kept only as a comparison feature.
- Use moisture indices too, mainly **NDMI** ((B8A - B11) / (B8A + B11)).
- Always mask clouds, cloud shadows, and saturated pixels with SCL before computing indices.
- Outputs must be tiny. Herders have very low bandwidth. Prefer:
  - vector data (simplified GeoJSON polygons or points, coordinates rounded to ~4 decimals), and
  - short SMS/USSD alert text (160 characters max, plain ASCII, French and Hassaniya Arabic transliteration).
  Never ship raster tiles to the mobile client.

## Processing & Modelling Constraints

- **Surveillance Task:** Calculate current SAVI/NDMI directly from the latest 5-day masked composites. No ML prediction applied here.
- **Forecasting Task:** Regression of pasture quality (biomass or a nutrition score) using LightGBM on historical index trajectories and season integrals.
- Validate ML models with spatial or temporal splits (e.g., GroupKFold on coordinates), never a random pixel split, to avoid leakage between neighbouring pixels.
- Report RMSE, MAE and R² per region and per season.

## Stack and conventions

- Python 3.11+. Suggested libraries: rasterio, xarray / rioxarray, geopandas, shapely, scikit-learn or LightGBM, pystac-client for imagery search.
- Suggested layout:
  - `pipeline/surveillance.py` 5-day near real-time STAC query, masking, and current index computation.
  - `model/seasonal.py` ML training, temporal feature extraction, evaluation, and seasonal inference.
  - `api/` backend that serves vector results and builds SMS/USSD messages.
  - `tests/` pytest tests.
  - `notebooks/` exploration only, no production logic.
- Keep large data out of git (`data/` is git-ignored).
- Run tests with `pytest` before committing.
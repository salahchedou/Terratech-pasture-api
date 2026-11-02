# Terratech Pasture API: project summary

_Last updated: 2026-11-02_

## Goal

Help cattle herders in southern Mauritania find nutritious pasture. Sentinel-2
satellite imagery is turned into pasture indicators and sent to herders in a
very lightweight form (small GeoJSON, SMS/USSD text), since bandwidth is low.

Area: the Sahelian pastoral belt (Hodh Ech Chargui, Hodh El Gharbi, Assaba,
Gorgol, Brakna, Trarza, Guidimakha). Rainy season July–October, dry-season
decline November–June.

## Architecture: two systems

| | 5-Day Surveillance Tracker | Seasonal AI Predictor |
|---|---|---|
| Question | What does the pasture look like **now**? | How much pasture will be **left** in the dry season? |
| Input | Scenes from the last 5 days over one grid (default 50 × 50 km) | Rainy-season SAVI/NDMI time series + labels |
| Method | Masking + per-pixel median SAVI/NDMI, **no ML** | LightGBM regression on season features |
| Code | `pipeline/surveillance.py` | `model/seasonal.py` |
| Output | GeoJSON points, one per cell (default 5 km) | Biomass / nutrition score per point |

## What is built

### `pipeline/ingest.py`: imagery search
- Searches Sentinel-2 L2A on Earth Search (AWS) by wilaya name or bbox.
- Approximate bounding boxes for the 7 southern wilayas.
- Returns, per scene, the links to bands B02, B03, B04, B08, B8A, B11, B12 and SCL.

### `pipeline/preprocess.py`: masking and indices
- Reads only the needed window from each cloud-optimised GeoTIFF.
- `Grid` / `make_grid`: one shared pixel grid, so scenes from different UTM
  zones (28, 29, 30 N) can be stacked; reprojection is done on the fly.
- Converts DN to reflectance (`DN × 0.0001 − 0.1`).
- SCL mask: drops no-data, saturated, dark, cloud shadow, clouds, cirrus, water, snow.
- Indices: **SAVI** (L = 0.5, main), **NDMI** ((B8A − B11)/(B8A + B11)), NDVI (comparison only).
- `assess_scene`: usability report (clear fraction, impossible reflectance, flat images).
- HTTP retries for truncated reads from the public S3 bucket.

### `pipeline/surveillance.py`: 5-day tracker
- `square_bbox`: box of N km around a point. `cycle_dates`: the 5-day window.
- Picks the UTM zone of the grid centre, loads every scene in the window,
  skips unreadable or broken scenes, and takes the per-pixel median.
- `cell_summary` + `to_geojson`: one point per cell with clear fraction, median
  SAVI and NDMI; coordinates rounded to 4 decimals.
- Command line:
  ```
  python -m pipeline.surveillance --lon -11.40 --lat 16.62 --size-km 5 --cell-km 1 \
      --end 2025-10-15 --output data/surveillance/kiffa_5km.geojson
  ```

### `model/seasonal.py`: seasonal predictor
- `season_of`, `season_year`: July 2025–June 2026 = season 2025.
- `rainy_season_features`: per point and season: mean, max, min, end value,
  integral, peak day, Sept–Oct slope (for SAVI and NDMI), NDVI max/mean.
- `spatial_groups`: ~20 km blocks for **GroupKFold**, so neighbouring pixels
  never sit on both sides of a split. `temporal_split`: train on past years only.
- `cross_validate`, `train`, `predict` with LightGBM.
- `metrics_by`: RMSE, MAE, R² overall, per region and per season.

### `notebooks/explore_first_scene.py`: exploration
- Rainy vs dry-season comparison over a small box west of Kiffa.

### Tests
- 38 offline pytest tests (`tests/`), synthetic data, no network: `pytest`.

## Results so far

- Earth Search L2A needs the −0.1 reflectance offset (processing baseline ≥ 04.00).
- Over dusty Sahelian scenes, atmospheric correction over-corrects blue
  (≈42 % of B02 pixels negative in Oct 2025), so B02/B03 are not core bands.
- The SCL "vegetation" class covers only ~0.7 % of pixels and is useless as a label.
- Median SAVI near Kiffa: **0.11 in May 2025 → 0.18 in Oct 2025**.
- Live surveillance run, 5 km box around Kiffa, 11–15 Oct 2025: 2 scenes
  (tiles 29QKU and 28QHD, two UTM zones), 89 % clear pixels, median SAVI 0.178,
  NDMI −0.154; 36 points in a 4.9 KB GeoJSON.

## Known limitations

- Wilaya boxes are approximate; official boundaries are still needed.
- A full 50 km run takes several minutes and a lot of memory.
- Surveillance outputs raw index values; "good / poor pasture" thresholds need field data.
- **No ground-truth labels yet**: the seasonal model is only tested on synthetic data.

## Next steps

1. Get ground-truth labels (field biomass or a nutrition score).
2. Build per-point SAVI/NDMI time series over past rainy seasons to train the predictor.
3. `api/`: serve the GeoJSON and build SMS/USSD alerts (≤ 160 chars, ASCII,
   French + Hassaniya transliteration).
4. Replace approximate boxes with official wilaya boundaries.
5. Schedule the tracker every 5 days for the chosen grids.

## Setup

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
pytest
```

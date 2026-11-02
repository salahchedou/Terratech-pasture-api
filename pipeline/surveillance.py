"""5-day surveillance tracker: current pasture state for one grid, no ML.

Every Sentinel-2 revisit (~5 days) this pulls the scenes acquired over a grid
(by default a 50 x 50 km box), masks them with SCL, composites SAVI and NDMI
per pixel, and summarises the result as a handful of GeoJSON points small
enough to send over a very slow connection.

    from pipeline.surveillance import square_bbox, run_surveillance
    bbox = square_bbox(-11.40, 16.62, size_km=50)        # around Kiffa
    report = run_surveillance(bbox, end="2025-10-15")
    report["geojson"]                                    # tiny vector output

Command line:

    python -m pipeline.surveillance --lon -11.40 --lat 16.62 --end 2025-10-15 \
        --output data/surveillance/kiffa_2025-10-15.geojson
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from datetime import date, timedelta
from typing import Any, Mapping, Sequence

import numpy as np

from pipeline.ingest import search_scenes, validate_bbox
from pipeline.preprocess import (
    CORE_BANDS,
    MAX_CORE_OUT_OF_RANGE,
    compute_indices,
    load_window,
    make_grid,
    out_of_range_fraction,
    scl_mask,
)

CYCLE_DAYS = 5  # Sentinel-2 revisit with two satellites
DEFAULT_GRID_KM = 50.0
# 20 m keeps a 50 km box at 2500 x 2500 pixels per band (B8A/B11 are 20 m natively).
DEFAULT_RESOLUTION = 20.0
DEFAULT_CELL_KM = 5.0  # one output point per 5 x 5 km cell -> 100 points per grid
MIN_CELL_VALID = 0.3  # skip cells with less than 30% clear pasture pixels
# Scene-level cloud filter is loose: SCL masks clouds per pixel, and a partly
# cloudy scene is still useful for the clear part of the grid.
DEFAULT_MAX_CLOUD = 60.0
SUMMARY_INDICES = ("SAVI", "NDMI")
COORD_DECIMALS = 4
VALUE_DECIMALS = 3


# ---------------------------------------------------------------------------
# Small helpers (pure, tested offline)
# ---------------------------------------------------------------------------


def square_bbox(lon: float, lat: float, size_km: float = DEFAULT_GRID_KM) -> tuple[float, float, float, float]:
    """Lon/lat box of roughly ``size_km`` x ``size_km`` centred on a point."""
    half_lat = (size_km / 2) / 111.32
    half_lon = (size_km / 2) / (111.32 * math.cos(math.radians(lat)))
    return validate_bbox((lon - half_lon, lat - half_lat, lon + half_lon, lat + half_lat))


def cycle_dates(end: str | date | None = None, days: int = CYCLE_DAYS) -> tuple[str, str]:
    """(start, end) ISO dates of the ``days``-long cycle ending on ``end`` (inclusive)."""
    if end is None:
        end_d = date.today()
    elif isinstance(end, date):
        end_d = end
    else:
        end_d = date.fromisoformat(end)
    return (end_d - timedelta(days=days - 1)).isoformat(), end_d.isoformat()


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 UTM zone containing a point (Mauritania: 28-30 N)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def composite(stack: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Per-pixel median of each index across scenes, ignoring masked (NaN) pixels.

    Also returns ``n_obs``: how many clear observations each pixel had.
    """
    if not stack:
        raise ValueError("nothing to composite")
    out: dict[str, np.ndarray] = {}
    for name in stack[0]:
        cube = np.stack([s[name] for s in stack])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            out[name] = np.nanmedian(cube, axis=0).astype(np.float32)
    first = next(iter(stack[0]))
    out["n_obs"] = np.sum([np.isfinite(s[first]) for s in stack], axis=0).astype(np.uint8)
    return out


def cell_summary(
    indices: Mapping[str, np.ndarray],
    cell_px: int,
    *,
    names: Sequence[str] = SUMMARY_INDICES,
    min_valid: float = MIN_CELL_VALID,
) -> list[dict[str, Any]]:
    """Aggregate pixel indices into square cells of ``cell_px`` pixels.

    Returns one record per cell with enough clear pixels: its centre in pixel
    coordinates (row, col, may be fractional), the clear fraction and the
    median of each index.
    """
    ref = indices[names[0]]
    h, w = ref.shape
    records = []
    for r0 in range(0, h, cell_px):
        for c0 in range(0, w, cell_px):
            block = ref[r0:r0 + cell_px, c0:c0 + cell_px]
            valid = np.isfinite(block)
            frac = float(valid.mean())
            if frac < min_valid:
                continue
            rec: dict[str, Any] = {
                "row": r0 + block.shape[0] / 2,
                "col": c0 + block.shape[1] / 2,
                "valid": round(frac, 2),
            }
            for name in names:
                v = indices[name][r0:r0 + cell_px, c0:c0 + cell_px]
                v = v[np.isfinite(v)]
                rec[name.lower()] = round(float(np.median(v)), VALUE_DECIMALS) if v.size else None
            records.append(rec)
    return records


def to_geojson(records: Sequence[Mapping[str, Any]], lons: Sequence[float], lats: Sequence[float],
               properties: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Point FeatureCollection, coordinates rounded to ~10 m (4 decimals)."""
    features = []
    for rec, lon, lat in zip(records, lons, lats):
        props = {k: v for k, v in rec.items() if k not in ("row", "col")}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(float(lon), COORD_DECIMALS), round(float(lat), COORD_DECIMALS)]},
            "properties": props,
        })
    fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if properties:
        fc["properties"] = dict(properties)
    return fc


def _scene_ok(window) -> bool:
    """Reject scenes whose core bands are broken on clear pixels.

    Partial coverage is fine here (a tile may only touch part of the grid),
    so the strict valid-fraction rule of assess_scene is not applied.
    """
    oor = out_of_range_fraction(
        {b: window.reflectance[b] for b in CORE_BANDS}, scl_mask(window.scl)
    )
    return window.mask.any() and all(f <= MAX_CORE_OUT_OF_RANGE for f in oor.values())


# ---------------------------------------------------------------------------
# Main entry point (network)
# ---------------------------------------------------------------------------


def run_surveillance(
    bbox: Sequence[float],
    end: str | date | None = None,
    *,
    days: int = CYCLE_DAYS,
    resolution: float = DEFAULT_RESOLUTION,
    cell_km: float = DEFAULT_CELL_KM,
    max_cloud: float | None = DEFAULT_MAX_CLOUD,
    catalog: Any = None,
) -> dict[str, Any]:
    """Current SAVI/NDMI for a grid from the latest ``days``-day cycle.

    Returns a dict with the cycle dates, the scenes used or skipped, overall
    index stats and ``geojson``: one point per cell with median SAVI/NDMI.
    """
    bbox = validate_bbox(bbox)
    start, end_s = cycle_dates(end, days)
    lon_c, lat_c = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    crs = f"EPSG:{utm_epsg(lon_c, lat_c)}"
    grid = make_grid(bbox, crs, resolution)

    scenes = search_scenes(bbox, start, end_s, max_cloud=max_cloud, catalog=catalog)
    used, skipped, stack = [], [], []
    for scene in scenes:
        try:
            window = load_window(scene["assets"], grid=grid)
        except Exception as exc:  # network / corrupt tile: skip, don't lose the cycle
            skipped.append({"id": scene["id"], "reason": f"read failed: {exc}"[:200]})
            continue
        if not _scene_ok(window):
            skipped.append({"id": scene["id"], "reason": "no clear pixels or broken core bands"})
            continue
        idx = compute_indices(window)
        stack.append({k: idx[k] for k in SUMMARY_INDICES})
        used.append(scene["id"])
        del window  # reflectance is large; keep only the indices

    report: dict[str, Any] = {
        "bbox": [round(v, COORD_DECIMALS) for v in bbox],
        "start": start,
        "end": end_s,
        "scenes_used": used,
        "scenes_skipped": skipped,
    }
    if not stack:
        report.update(clear_fraction=0.0, geojson=to_geojson([], [], [], {"start": start, "end": end_s}))
        return report

    comp = composite(stack)
    clear = np.isfinite(comp["SAVI"])
    report["clear_fraction"] = round(float(clear.mean()), 4)
    report["median"] = {
        k.lower(): round(float(np.nanmedian(comp[k])), VALUE_DECIMALS) if clear.any() else None
        for k in SUMMARY_INDICES
    }

    cell_px = max(1, int(round(cell_km * 1000 / resolution)))
    records = cell_summary(comp, cell_px)
    lons, lats = _pixel_centres_to_lonlat(records, grid)
    report["geojson"] = to_geojson(records, lons, lats, {"start": start, "end": end_s})
    return report


def _pixel_centres_to_lonlat(records, grid) -> tuple[list[float], list[float]]:
    from rasterio.warp import transform

    xs, ys = [], []
    for rec in records:
        x, y = grid.transform * (rec["col"], rec["row"])
        xs.append(x)
        ys.append(y)
    if not xs:
        return [], []
    lons, lats = transform(grid.crs, "EPSG:4326", xs, ys)
    return list(lons), list(lats)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--lon", type=float, required=True, help="grid centre longitude")
    p.add_argument("--lat", type=float, required=True, help="grid centre latitude")
    p.add_argument("--size-km", type=float, default=DEFAULT_GRID_KM)
    p.add_argument("--end", help="last day of the cycle, YYYY-MM-DD (default: today)")
    p.add_argument("--days", type=int, default=CYCLE_DAYS)
    p.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    p.add_argument("--cell-km", type=float, default=DEFAULT_CELL_KM)
    p.add_argument("--output", help="write the GeoJSON here (the full report goes to stderr)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_surveillance(
        square_bbox(args.lon, args.lat, args.size_km),
        args.end,
        days=args.days,
        resolution=args.resolution,
        cell_km=args.cell_km,
    )
    geojson = json.dumps(report["geojson"], separators=(",", ":"))
    summary = {k: v for k, v in report.items() if k != "geojson"}
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(geojson)
        print(json.dumps(summary, indent=2), file=sys.stderr)
        print(f"{len(report['geojson']['features'])} points, {len(geojson)} bytes -> {args.output}",
              file=sys.stderr)
    else:
        print(geojson)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

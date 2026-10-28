"""Query Sentinel-2 Level-2A imagery for southern Mauritania via STAC.

Usage as a library:

    from pipeline.ingest import search_wilaya
    scenes = search_wilaya("Assaba", "2025-08-01", "2025-09-30", max_cloud=30)
    for s in scenes:
        print(s["id"], s["assets"]["B8A"])

Usage from the command line:

    python -m pipeline.ingest --wilaya Assaba --start 2025-08-01 --end 2025-09-30
    python -m pipeline.ingest --bbox -12.9 15.6 -10.6 17.6 --start 2025-08-01 --end 2025-08-31
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Iterable, Sequence

# (min_lon, min_lat, max_lon, max_lat) in EPSG:4326
BBox = tuple[float, float, float, float]

DEFAULT_CATALOG_URL = "https://earth-search.aws.element84.com/v1"
DEFAULT_COLLECTION = "sentinel-2-l2a"

# Bands needed downstream: SAVI/NDVI (B04, B08), NDMI (B8A, B11),
# extra spectral features (B02, B03, B12) and SCL for cloud/shadow masking.
BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12", "SCL")

# Catalogs name assets differently. Earth Search v1 uses common names,
# Planetary Computer uses the band IDs directly. We try the band ID first,
# then these aliases.
ASSET_ALIASES: dict[str, tuple[str, ...]] = {
    "B02": ("blue",),
    "B03": ("green",),
    "B04": ("red",),
    "B08": ("nir",),
    "B8A": ("nir08",),
    "B11": ("swir16",),
    "B12": ("swir22",),
    "SCL": ("scl",),
}

# APPROXIMATE bounding boxes for the southern pastoral wilayas, clipped to the
# Sahelian belt (Hodh Ech Chargui extends much further north into desert).
# Good enough for imagery search; replace with official boundaries
# (e.g. geoBoundaries / GADM polygons) for any clipping or reporting.
WILAYA_BBOXES: dict[str, BBox] = {
    "hodh_ech_chargui": (-8.0, 15.3, -5.3, 17.8),
    "hodh_el_gharbi": (-10.9, 15.5, -8.0, 17.5),
    "assaba": (-12.9, 15.6, -10.6, 17.6),
    "gorgol": (-13.9, 15.3, -11.7, 16.8),
    "brakna": (-14.8, 16.4, -12.5, 18.4),
    "trarza": (-16.6, 16.3, -14.2, 18.3),
    "guidimakha": (-12.9, 14.7, -11.3, 15.8),
}

# Whole southern pastoral belt (union of the boxes above).
SOUTHERN_BELT_BBOX: BBox = (-16.6, 14.7, -5.3, 18.4)


def _normalise_name(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("'", "")
    )


def get_wilaya_bbox(name: str) -> BBox:
    """Return the bounding box for a wilaya name (case and separator insensitive)."""
    key = _normalise_name(name)
    if key in ("southern_belt", "south", "all"):
        return SOUTHERN_BELT_BBOX
    try:
        return WILAYA_BBOXES[key]
    except KeyError:
        known = ", ".join(sorted(WILAYA_BBOXES))
        raise ValueError(f"Unknown wilaya {name!r}. Known: {known}") from None


def validate_bbox(bbox: Sequence[float]) -> BBox:
    """Check a bbox is (min_lon, min_lat, max_lon, max_lat) in valid ranges."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(bbox)}")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bbox)
    if not (-180 <= min_lon < max_lon <= 180):
        raise ValueError(f"invalid longitudes: {min_lon}, {max_lon}")
    if not (-90 <= min_lat < max_lat <= 90):
        raise ValueError(f"invalid latitudes: {min_lat}, {max_lat}")
    return (min_lon, min_lat, max_lon, max_lat)


def open_catalog(url: str = DEFAULT_CATALOG_URL):
    """Open a STAC API catalog."""
    from pystac_client import Client  # imported lazily so helpers work offline

    return Client.open(url)


def resolve_asset_key(assets: Iterable[str], band: str) -> str | None:
    """Find the asset key for a band in an item's assets, or None."""
    keys = set(assets)
    for candidate in (band, *ASSET_ALIASES.get(band, ())):
        if candidate in keys:
            return candidate
    return None


def select_band_assets(item: Any, bands: Sequence[str] = BANDS) -> dict[str, str]:
    """Return {band: href} for the requested bands of a STAC item.

    Raises KeyError if any requested band is missing.
    """
    hrefs: dict[str, str] = {}
    missing: list[str] = []
    for band in bands:
        key = resolve_asset_key(item.assets, band)
        if key is None:
            missing.append(band)
        else:
            hrefs[band] = item.assets[key].href
    if missing:
        raise KeyError(f"item {item.id} is missing bands: {missing}")
    return hrefs


def search_scenes(
    bbox: Sequence[float],
    start: str,
    end: str,
    *,
    max_cloud: float | None = 40.0,
    bands: Sequence[str] = BANDS,
    catalog_url: str = DEFAULT_CATALOG_URL,
    collection: str = DEFAULT_COLLECTION,
    max_items: int | None = 200,
    catalog: Any = None,
) -> list[dict[str, Any]]:
    """Search Sentinel-2 L2A scenes intersecting ``bbox`` between two dates.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        start, end: ISO dates, e.g. "2025-08-01". Both inclusive.
        max_cloud: scene-level cloud cover filter in percent (None to disable).
            Pixel-level masking with SCL still has to happen downstream.
        bands: bands to keep in the result.
        catalog_url, collection: STAC API endpoint and collection id.
        max_items: cap on returned scenes (None for no cap).
        catalog: an already-open pystac_client.Client (handy for tests/reuse).

    Returns:
        A list of dicts, oldest first, each with id, datetime, cloud_cover,
        mgrs_tile, bbox and an ``assets`` mapping {band: href}. Scenes missing
        any requested band are skipped.
    """
    bbox = validate_bbox(bbox)
    client = catalog if catalog is not None else open_catalog(catalog_url)

    query = {"eo:cloud_cover": {"lt": max_cloud}} if max_cloud is not None else None
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query=query,
        max_items=max_items,
    )

    scenes: list[dict[str, Any]] = []
    for item in search.items():
        try:
            assets = select_band_assets(item, bands)
        except KeyError:
            continue
        props = item.properties
        scenes.append(
            {
                "id": item.id,
                "datetime": props.get("datetime"),
                "cloud_cover": props.get("eo:cloud_cover"),
                "mgrs_tile": props.get("s2:mgrs_tile") or props.get("grid:code"),
                "bbox": item.bbox,
                "assets": assets,
            }
        )
    scenes.sort(key=lambda s: s["datetime"] or "")
    return scenes


def search_wilaya(wilaya: str, start: str, end: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Search scenes for a named wilaya. Extra kwargs go to search_scenes."""
    return search_scenes(get_wilaya_bbox(wilaya), start, end, **kwargs)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--wilaya", help=f"one of: {', '.join(WILAYA_BBOXES)}, southern_belt")
    where.add_argument(
        "--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT")
    )
    p.add_argument("--start", required=True, help="start date, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="end date, YYYY-MM-DD")
    p.add_argument("--max-cloud", type=float, default=40.0, help="max scene cloud %% (default 40)")
    p.add_argument("--max-items", type=int, default=200)
    p.add_argument("--catalog-url", default=DEFAULT_CATALOG_URL)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--output", help="write results as JSON to this file instead of stdout")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bbox = get_wilaya_bbox(args.wilaya) if args.wilaya else validate_bbox(args.bbox)
    scenes = search_scenes(
        bbox,
        args.start,
        args.end,
        max_cloud=args.max_cloud,
        max_items=args.max_items,
        catalog_url=args.catalog_url,
        collection=args.collection,
    )
    text = json.dumps(scenes, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{len(scenes)} scenes written to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

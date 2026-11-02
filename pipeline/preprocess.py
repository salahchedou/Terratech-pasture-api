"""Turn a Sentinel-2 L2A scene into masked reflectance and vegetation indices.

Steps, in order:
1. Read a small window (lon/lat box) out of each band's cloud-optimised GeoTIFF,
   resampled onto one common 10 m grid. Only that window is downloaded.
2. Convert digital numbers to surface reflectance (DN * scale + offset).
3. Mask no-data, saturated/defective, dark, cloud shadow, cloud and cirrus pixels
   with the SCL layer, plus water and snow, which are not pasture.
4. Compute SAVI (L = 0.5, main index), NDMI (moisture) and NDVI (comparison only).
5. Score the window so we can decide if the scene is usable for the model.

Example:

    from pipeline.preprocess import load_window, compute_indices, assess_scene
    data = load_window(scene["assets"], bbox=(-11.5, 16.5, -11.3, 16.7))
    idx = compute_indices(data)
    report = assess_scene(data, idx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

# Sentinel-2 SCL (scene classification) values.
SCL_CLASSES: dict[int, str] = {
    0: "no_data",
    1: "saturated_or_defective",
    2: "dark_area",  # mostly topographic / cast shadows
    3: "cloud_shadow",
    4: "vegetation",
    5: "not_vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud_medium_probability",
    9: "cloud_high_probability",
    10: "thin_cirrus",
    11: "snow_or_ice",
}

# Pixels we never trust (bad data, clouds, shadows).
SCL_INVALID: frozenset[int] = frozenset({0, 1, 2, 3, 8, 9, 10})
# Pixels that are fine optically but are not pasture.
SCL_NOT_PASTURE: frozenset[int] = frozenset({6, 11})

# Default scaling for Earth Search L2A COGs (processing baseline >= 04.00).
# Real values are read from the STAC item when available.
DEFAULT_SCALE = 0.0001
DEFAULT_OFFSET = -0.1
NODATA_DN = 0
SATURATED_DN = 65535

SAVI_L = 0.5
REFLECTANCE_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08", "B8A", "B11", "B12")
# Bands the indices and model depend on: a pixel is dropped if any of these is
# missing or outside [0, 1]. Blue/green are not core: over dusty Sahelian
# scenes Sen2Cor often over-corrects them for aerosols, pushing B02 below 0.
CORE_BANDS: tuple[str, ...] = ("B04", "B08", "B8A", "B11", "B12")


@dataclass
class SceneWindow:
    """Reflectance bands and SCL on one 10 m grid."""

    reflectance: dict[str, np.ndarray]  # float32, NaN where unusable
    scl: np.ndarray  # uint8
    mask: np.ndarray  # bool, True = usable pasture pixel
    transform: Any = None  # affine transform of the grid
    crs: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure array functions (no I/O, easy to test)
# ---------------------------------------------------------------------------


def dn_to_reflectance(
    dn: np.ndarray, scale: float = DEFAULT_SCALE, offset: float = DEFAULT_OFFSET
) -> np.ndarray:
    """Convert digital numbers to surface reflectance.

    No-data (0) and saturated (65535) pixels become NaN.
    """
    dn = np.asarray(dn)
    refl = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    bad = (dn == NODATA_DN) | (dn == SATURATED_DN)
    refl[bad] = np.nan
    return refl


def scl_mask(scl: np.ndarray, exclude_not_pasture: bool = True) -> np.ndarray:
    """Return True where the SCL class is usable."""
    drop = set(SCL_INVALID)
    if exclude_not_pasture:
        drop |= SCL_NOT_PASTURE
    return ~np.isin(scl, list(drop))


def build_mask(
    reflectance: Mapping[str, np.ndarray],
    scl: np.ndarray,
    core_bands: Sequence[str] = CORE_BANDS,
) -> np.ndarray:
    """Combine SCL with per-band checks on the core bands.

    A pixel is kept only if every core band is present (not no-data or
    saturated) and physically plausible (0 <= reflectance <= 1).
    """
    mask = scl_mask(scl)
    for band in core_bands:
        if band in reflectance:
            arr = reflectance[band]
            with np.errstate(invalid="ignore"):
                mask &= np.isfinite(arr) & (arr >= 0) & (arr <= 1)
    return mask


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32)


def savi(nir: np.ndarray, red: np.ndarray, L: float = SAVI_L) -> np.ndarray:
    """Soil Adjusted Vegetation Index: (1 + L)(NIR - Red) / (NIR + Red + L)."""
    return _safe_ratio((1 + L) * (nir - red), nir + red + L)


def ndmi(nir08: np.ndarray, swir16: np.ndarray) -> np.ndarray:
    """Normalised Difference Moisture Index: (B8A - B11) / (B8A + B11)."""
    return _safe_ratio(nir08 - swir16, nir08 + swir16)


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """NDVI, kept only as a comparison feature."""
    return _safe_ratio(nir - red, nir + red)


def compute_indices(window: SceneWindow) -> dict[str, np.ndarray]:
    """SAVI, NDMI and NDVI for a window, NaN outside the mask."""
    r = window.reflectance
    out = {
        "SAVI": savi(r["B08"], r["B04"]),
        "NDMI": ndmi(r["B8A"], r["B11"]),
        "NDVI": ndvi(r["B08"], r["B04"]),
    }
    for arr in out.values():
        arr[~window.mask] = np.nan
    return out


# ---------------------------------------------------------------------------
# Usability check
# ---------------------------------------------------------------------------

# A window is "usable" for training/inference when it passes all of these.
MIN_VALID_FRACTION = 0.7  # at least 70% of pixels are clear pasture
MAX_CORE_OUT_OF_RANGE = 0.05  # core band outside [0, 1] on clear pixels
MIN_SAVI_STD = 0.01  # some spatial variation, not a flat/broken image
AUX_WARN_OUT_OF_RANGE = 0.05  # warn (only) when blue/green look over-corrected


def out_of_range_fraction(
    reflectance: Mapping[str, np.ndarray], clear: np.ndarray
) -> dict[str, float]:
    """Share of clear (SCL-valid, present) pixels outside [0, 1], per band."""
    out = {}
    for band, arr in reflectance.items():
        v = arr[clear & np.isfinite(arr)]
        out[band] = round(float(((v < 0) | (v > 1)).mean()), 4) if v.size else 1.0
    return out


def _stats(arr: np.ndarray) -> dict[str, float | None]:
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"min": None, "p05": None, "median": None, "mean": None, "p95": None, "max": None, "std": None}
    p05, med, p95 = np.percentile(v, [5, 50, 95])
    return {
        "min": round(float(v.min()), 4),
        "p05": round(float(p05), 4),
        "median": round(float(med), 4),
        "mean": round(float(v.mean()), 4),
        "p95": round(float(p95), 4),
        "max": round(float(v.max()), 4),
        "std": round(float(v.std()), 4),
    }


def assess_scene(window: SceneWindow, indices: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Summarise a window and decide whether it is usable for the model."""
    total = window.scl.size
    classes, counts = np.unique(window.scl, return_counts=True)
    scl_share = {
        SCL_CLASSES.get(int(c), str(int(c))): round(int(n) / total, 4)
        for c, n in zip(classes, counts)
    }
    valid_fraction = float(window.mask.sum()) / total if total else 0.0

    # Physically impossible reflectance on otherwise clear pixels.
    oor = out_of_range_fraction(window.reflectance, scl_mask(window.scl))

    index_stats = {name: _stats(arr) for name, arr in indices.items()}
    savi_std = index_stats["SAVI"]["std"] or 0.0

    reasons, warnings = [], []
    if valid_fraction < MIN_VALID_FRACTION:
        reasons.append(f"only {valid_fraction:.0%} clear pasture pixels (< {MIN_VALID_FRACTION:.0%})")
    for band, frac in oor.items():
        if band in CORE_BANDS and frac > MAX_CORE_OUT_OF_RANGE:
            reasons.append(f"{band}: {frac:.1%} of clear pixels outside [0, 1]")
        elif band not in CORE_BANDS and frac > AUX_WARN_OUT_OF_RANGE:
            warnings.append(f"{band}: {frac:.1%} outside [0, 1] (likely aerosol over-correction); "
                            "do not use as a feature for this scene")
    if savi_std < MIN_SAVI_STD:
        reasons.append(f"SAVI almost constant (std {savi_std:.3f})")

    return {
        "usable": not reasons,
        "reasons": reasons,
        "warnings": warnings,
        "pixels": int(total),
        "valid_fraction": round(valid_fraction, 4),
        "out_of_range_fraction": oor,
        "scl_share": scl_share,
        "indices": index_stats,
    }


# ---------------------------------------------------------------------------
# Reading from the cloud-optimised GeoTIFFs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A north-up pixel grid: where every band and every scene is resampled to."""

    crs: Any
    transform: Any
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        res = self.transform.a
        return (left, top - self.height * res, left + self.width * res, top)


def make_grid(bbox: Sequence[float], crs: Any, resolution: float = 10.0) -> Grid:
    """Grid covering a lon/lat bbox in ``crs``, snapped to multiples of ``resolution``."""
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    left, bottom, right, top = transform_bounds("EPSG:4326", crs, *bbox)
    left = np.floor(left / resolution) * resolution
    top = np.ceil(top / resolution) * resolution
    width = int(np.ceil((right - left) / resolution))
    height = int(np.ceil((top - bottom) / resolution))
    return Grid(crs=crs, transform=from_origin(left, top, resolution, resolution),
                width=width, height=height)


def scene_crs(assets: Mapping[str, str]) -> Any:
    """CRS of a scene (read from the SCL header only)."""
    import rasterio

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES"):
        with rasterio.open(assets["SCL"]) as src:
            return src.crs


def load_window(
    assets: Mapping[str, str],
    bbox: Sequence[float] | None = None,
    *,
    grid: Grid | None = None,
    resolution: float = 10.0,
    scale: float = DEFAULT_SCALE,
    offset: float = DEFAULT_OFFSET,
    bands: Sequence[str] = REFLECTANCE_BANDS,
) -> SceneWindow:
    """Read a lon/lat box from each band onto a common grid and mask it.

    Args:
        assets: {band: href}, as returned by pipeline.ingest.search_scenes.
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326. Used to build
            a grid in the scene's own CRS when ``grid`` is not given.
        grid: target grid. Pass the same grid for several scenes (even from
            different UTM zones) to stack or composite them pixel by pixel.
        resolution: pixel size in metres when building the grid from ``bbox``.
        scale, offset: DN to reflectance conversion (see the item's raster:bands).
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    if grid is None:
        if bbox is None:
            raise ValueError("give either bbox or grid")
        grid = make_grid(bbox, scene_crs(assets), resolution)
    left, bottom, right, top = grid.bounds

    env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_HTTP_MULTIRANGE="YES",
        # Public S3 reads sometimes come back truncated; retry before failing.
        GDAL_HTTP_MAX_RETRY="5",
        GDAL_HTTP_RETRY_DELAY="2",
    )

    def _read(href: str) -> np.ndarray:
        # Nearest neighbour everywhere: SCL is a class map, and for reflectance
        # it keeps no-data/saturated DN from being blended into good pixels.
        with rasterio.open(href) as src:
            if src.crs == grid.crs:
                win = from_bounds(left, bottom, right, top, transform=src.transform)
                return src.read(1, window=win, out_shape=grid.shape,
                                resampling=Resampling.nearest,
                                boundless=True, fill_value=NODATA_DN)
            # Different UTM zone: warp on the fly, only the needed blocks are fetched.
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform,
                           width=grid.width, height=grid.height,
                           resampling=Resampling.nearest,
                           src_nodata=NODATA_DN, nodata=NODATA_DN) as vrt:
                return vrt.read(1)

    with env:
        scl = _read(assets["SCL"]).astype(np.uint8)
        reflectance = {b: dn_to_reflectance(_read(assets[b]), scale, offset) for b in bands}

    return SceneWindow(
        reflectance=reflectance,
        scl=scl,
        mask=build_mask(reflectance, scl),
        transform=grid.transform,
        crs=grid.crs,
        meta={"bbox_wgs84": tuple(bbox) if bbox is not None else None,
              "resolution": grid.transform.a},
    )

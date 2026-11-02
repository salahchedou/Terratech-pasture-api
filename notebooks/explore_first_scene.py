"""Exploration: run search -> mask -> indices -> usability check on a real window.

Not production code. Compares a rainy-season and a dry-season scene over the
same small box of rangeland near Kiffa (Assaba) and writes quicklook PNGs and a
JSON report to data/explore/ (git-ignored).

    python -m notebooks.explore_first_scene
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.ingest import search_scenes
from pipeline.preprocess import assess_scene, compute_indices, load_window

# ~10 x 11 km of rangeland just west of Kiffa.
BBOX = (-11.55, 16.55, -11.45, 16.65)
SEASONS = {
    "rainy_end": ("2025-09-25", "2025-10-20"),
    "dry_late": ("2025-04-15", "2025-05-31"),
}
OUT = Path("data/explore")


def pick_scene(start: str, end: str) -> dict:
    """Lowest-cloud scene whose tile fully covers BBOX (skip empty tile edges)."""
    scenes = search_scenes(BBOX, start, end, max_cloud=20, max_items=100)
    lon0, lat0, lon1, lat1 = BBOX
    covering = [
        s for s in scenes
        if s["bbox"][0] <= lon0 and s["bbox"][1] <= lat0 and s["bbox"][2] >= lon1 and s["bbox"][3] >= lat1
    ]
    if not covering:
        raise RuntimeError(f"no scene covers {BBOX} between {start} and {end}")
    return min(covering, key=lambda s: s["cloud_cover"])


def to_png(path: Path, rgb: np.ndarray) -> None:
    import warnings

    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    warnings.simplefilter("ignore", NotGeoreferencedWarning)  # PNGs are just previews
    with rasterio.open(path, "w", driver="PNG", width=rgb.shape[2], height=rgb.shape[1],
                       count=3, dtype="uint8") as dst:
        dst.write(rgb)


def true_colour(window) -> np.ndarray:
    r = window.reflectance
    rgb = np.stack([r["B04"], r["B03"], r["B02"]])
    rgb = np.clip(np.nan_to_num(rgb) / 0.35, 0, 1)  # bright sandy soils
    return (rgb * 255).astype(np.uint8)


def savi_colour(savi: np.ndarray, vmin: float = 0.0, vmax: float = 0.5) -> np.ndarray:
    """Brown (bare) -> yellow -> green (dense). Masked pixels are grey."""
    t = np.clip((savi - vmin) / (vmax - vmin), 0, 1)
    stops = np.array([[140, 90, 40], [230, 210, 90], [30, 130, 40]], dtype=np.float32)
    lo = np.where(t[..., None] < 0.5, stops[0], stops[1])
    hi = np.where(t[..., None] < 0.5, stops[1], stops[2])
    f = np.where(t < 0.5, t * 2, (t - 0.5) * 2)[..., None]
    rgb = lo + (hi - lo) * f
    rgb[~np.isfinite(savi)] = 128
    return rgb.transpose(2, 0, 1).astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for season, (start, end) in SEASONS.items():
        scene = pick_scene(start, end)
        print(f"[{season}] {scene['id']}  cloud {scene['cloud_cover']:.2f}%")
        window = load_window(scene["assets"], BBOX)
        idx = compute_indices(window)
        report = assess_scene(window, idx)
        report["scene_id"] = scene["id"]
        report["date"] = scene["datetime"][:10]
        reports[season] = report

        to_png(OUT / f"{season}_truecolour.png", true_colour(window))
        to_png(OUT / f"{season}_savi.png", savi_colour(idx["SAVI"]))
        print(json.dumps(report, indent=2))

    (OUT / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    wet, dry = reports["rainy_end"]["indices"], reports["dry_late"]["indices"]
    print("\nSeason contrast (median): "
          f"SAVI {dry['SAVI']['median']} -> {wet['SAVI']['median']}, "
          f"NDMI {dry['NDMI']['median']} -> {wet['NDMI']['median']}")


if __name__ == "__main__":
    main()

"""Offline tests for pipeline.surveillance (no network)."""

import math

import numpy as np
import pytest

from pipeline import surveillance as sv


def test_square_bbox_is_about_50_km():
    lon0, lat0, lon1, lat1 = sv.square_bbox(-11.4, 16.6, 50)
    assert (lat1 - lat0) * 111.32 == pytest.approx(50, rel=1e-6)
    assert (lon1 - lon0) * 111.32 * math.cos(math.radians(16.6)) == pytest.approx(50, rel=1e-6)


def test_cycle_dates_cover_five_days_inclusive():
    assert sv.cycle_dates("2025-10-15") == ("2025-10-11", "2025-10-15")


def test_utm_epsg_for_mauritania():
    assert sv.utm_epsg(-11.4, 16.6) == 32629  # Kiffa
    assert sv.utm_epsg(-15.9, 18.1) == 32628  # Nouakchott
    assert sv.utm_epsg(-5.8, 16.5) == 32630  # far east of Hodh Ech Chargui


def test_composite_median_ignores_masked_pixels():
    a = {"SAVI": np.array([[0.1, np.nan]], dtype=np.float32)}
    b = {"SAVI": np.array([[0.3, np.nan]], dtype=np.float32)}
    c = {"SAVI": np.array([[np.nan, np.nan]], dtype=np.float32)}
    out = sv.composite([a, b, c])
    assert out["SAVI"][0, 0] == pytest.approx(0.2)
    assert np.isnan(out["SAVI"][0, 1])
    assert out["n_obs"].tolist() == [[2, 0]]


def test_cell_summary_skips_cloudy_cells_and_takes_medians():
    savi = np.full((4, 4), 0.2, dtype=np.float32)
    ndmi = np.full((4, 4), -0.1, dtype=np.float32)
    savi[:2, 2:] = np.nan  # top-right cell fully masked
    recs = sv.cell_summary({"SAVI": savi, "NDMI": ndmi}, cell_px=2)
    assert len(recs) == 3
    assert {(r["row"], r["col"]) for r in recs} == {(1, 1), (3, 1), (3, 3)}
    assert all(r["savi"] == 0.2 and r["ndmi"] == -0.1 and r["valid"] == 1.0 for r in recs)


def test_geojson_is_small_and_rounded():
    recs = [{"row": 1, "col": 1, "valid": 0.9, "savi": 0.183, "ndmi": -0.121}]
    fc = sv.to_geojson(recs, [-11.4123456], [16.6198765], {"end": "2025-10-15"})
    feat = fc["features"][0]
    assert feat["geometry"]["coordinates"] == [-11.4123, 16.6199]
    assert "row" not in feat["properties"]
    assert fc["properties"]["end"] == "2025-10-15"


def test_run_surveillance_with_no_scenes_returns_empty_collection():
    class EmptyCatalog:
        def search(self, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(items=lambda: iter([]))

    report = sv.run_surveillance(sv.square_bbox(-11.4, 16.6), "2025-10-15", catalog=EmptyCatalog())
    assert report["scenes_used"] == []
    assert report["geojson"]["features"] == []
    assert report["start"] == "2025-10-11"

"""Offline tests for pipeline.preprocess (synthetic arrays, no network)."""

import numpy as np
import pytest

from pipeline import preprocess as pp


def test_dn_to_reflectance_applies_scale_and_offset():
    dn = np.array([1000, 3000, 11000], dtype=np.uint16)
    np.testing.assert_allclose(pp.dn_to_reflectance(dn), [0.0, 0.2, 1.0], atol=1e-6)


def test_dn_to_reflectance_nodata_and_saturated_become_nan():
    refl = pp.dn_to_reflectance(np.array([0, 65535, 2000], dtype=np.uint16))
    assert np.isnan(refl[0]) and np.isnan(refl[1])
    assert refl[2] == pytest.approx(0.1)


def test_scl_mask_drops_clouds_shadows_water_keeps_land():
    scl = np.arange(12, dtype=np.uint8)
    keep = pp.scl_mask(scl)
    assert set(np.flatnonzero(keep)) == {4, 5, 7}
    # Water and snow are kept when we are not filtering for pasture.
    assert set(np.flatnonzero(pp.scl_mask(scl, exclude_not_pasture=False))) == {4, 5, 6, 7, 11}


def test_savi_known_value():
    # (1.5 * (0.4 - 0.1)) / (0.4 + 0.1 + 0.5) = 0.45
    assert pp.savi(np.array([0.4]), np.array([0.1]))[0] == pytest.approx(0.45)


def test_ndmi_and_ndvi_known_values():
    assert pp.ndmi(np.array([0.3]), np.array([0.1]))[0] == pytest.approx(0.5)
    assert pp.ndvi(np.array([0.3]), np.array([0.1]))[0] == pytest.approx(0.5)


def test_index_zero_denominator_is_nan():
    assert np.isnan(pp.ndmi(np.array([0.0]), np.array([0.0]))[0])


def _window(n=100, blue=0.08, red=0.15, scl_value=5):
    shape = (n, n)
    refl = {b: np.full(shape, v, dtype=np.float32) for b, v in {
        "B02": blue, "B03": 0.1, "B04": red, "B08": 0.3,
        "B8A": 0.32, "B11": 0.45, "B12": 0.36,
    }.items()}
    # A bit of spatial variation so SAVI is not constant.
    refl["B08"] += np.linspace(0, 0.2, n, dtype=np.float32)[None, :]
    scl = np.full(shape, scl_value, dtype=np.uint8)
    return pp.SceneWindow(reflectance=refl, scl=scl, mask=pp.build_mask(refl, scl))


def test_build_mask_drops_out_of_range_core_band_only():
    w = _window(n=4)
    w.reflectance["B04"][0, 0] = -0.02  # core band negative -> drop
    w.reflectance["B02"][1, 1] = -0.05  # blue negative -> keep
    mask = pp.build_mask(w.reflectance, w.scl)
    assert not mask[0, 0]
    assert mask[1, 1]


def test_compute_indices_nan_outside_mask():
    w = _window(n=4)
    w.scl[0, 0] = 9  # cloud
    w.mask = pp.build_mask(w.reflectance, w.scl)
    idx = pp.compute_indices(w)
    assert np.isnan(idx["SAVI"][0, 0])
    assert np.isfinite(idx["SAVI"][1, 1])


def test_assess_clear_scene_is_usable():
    w = _window()
    report = pp.assess_scene(w, pp.compute_indices(w))
    assert report["usable"], report["reasons"]
    assert report["valid_fraction"] == 1.0
    assert report["warnings"] == []


def test_assess_cloudy_scene_is_not_usable():
    w = _window()
    w.scl[:60] = 9
    w.mask = pp.build_mask(w.reflectance, w.scl)
    report = pp.assess_scene(w, pp.compute_indices(w))
    assert not report["usable"]
    assert "clear pasture pixels" in report["reasons"][0]


def test_assess_negative_blue_only_warns():
    w = _window(blue=-0.01)
    report = pp.assess_scene(w, pp.compute_indices(w))
    assert report["usable"]
    assert report["warnings"] and report["warnings"][0].startswith("B02")


def test_assess_flat_savi_is_not_usable():
    w = _window()
    w.reflectance["B08"][:] = 0.3
    report = pp.assess_scene(w, pp.compute_indices(w))
    assert not report["usable"]
    assert any("SAVI almost constant" in r for r in report["reasons"])

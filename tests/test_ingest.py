"""Offline tests for pipeline.ingest (no network, no pystac-client needed)."""

from types import SimpleNamespace

import pytest

from pipeline import ingest


def _item(item_id, asset_keys, dt="2025-08-10T11:00:00Z", cloud=5.0):
    return SimpleNamespace(
        id=item_id,
        bbox=[-12.0, 16.0, -11.0, 17.0],
        properties={"datetime": dt, "eo:cloud_cover": cloud, "s2:mgrs_tile": "28QEC"},
        assets={k: SimpleNamespace(href=f"s3://bucket/{item_id}/{k}.tif") for k in asset_keys},
    )


EARTH_SEARCH_KEYS = ["blue", "green", "red", "nir", "nir08", "swir16", "swir22", "scl", "visual"]


class FakeCatalog:
    def __init__(self, items):
        self.items_ = items
        self.kwargs = None

    def search(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(items=lambda: iter(self.items_))


def test_wilaya_lookup_is_case_and_separator_insensitive():
    assert ingest.get_wilaya_bbox("Hodh Ech-Chargui") == ingest.WILAYA_BBOXES["hodh_ech_chargui"]
    assert ingest.get_wilaya_bbox("southern_belt") == ingest.SOUTHERN_BELT_BBOX


def test_unknown_wilaya_raises():
    with pytest.raises(ValueError, match="Unknown wilaya"):
        ingest.get_wilaya_bbox("Atlantis")


def test_all_wilaya_boxes_valid_and_inside_belt():
    b = ingest.SOUTHERN_BELT_BBOX
    for bbox in ingest.WILAYA_BBOXES.values():
        ingest.validate_bbox(bbox)
        assert b[0] <= bbox[0] and b[1] <= bbox[1] and bbox[2] <= b[2] and bbox[3] <= b[3]


@pytest.mark.parametrize("bad", [(1, 2, 3), (-10, 16, -12, 17), (-12, 18, -10, 16), (-200, 0, 0, 1)])
def test_validate_bbox_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ingest.validate_bbox(bad)


def test_select_band_assets_maps_common_names():
    hrefs = ingest.select_band_assets(_item("a", EARTH_SEARCH_KEYS))
    assert set(hrefs) == set(ingest.BANDS)
    assert hrefs["B8A"].endswith("/nir08.tif")
    assert hrefs["SCL"].endswith("/scl.tif")


def test_select_band_assets_accepts_band_ids():
    hrefs = ingest.select_band_assets(_item("b", list(ingest.BANDS)))
    assert hrefs["B11"].endswith("/B11.tif")


def test_select_band_assets_reports_missing():
    with pytest.raises(KeyError, match="SCL"):
        ingest.select_band_assets(_item("c", ["blue", "green", "red", "nir", "nir08", "swir16", "swir22"]))


def test_search_scenes_builds_query_and_skips_incomplete_items():
    items = [
        _item("late", EARTH_SEARCH_KEYS, dt="2025-08-20T11:00:00Z"),
        _item("no_scl", EARTH_SEARCH_KEYS[:7]),
        _item("early", EARTH_SEARCH_KEYS, dt="2025-08-05T11:00:00Z"),
    ]
    cat = FakeCatalog(items)
    scenes = ingest.search_wilaya("Assaba", "2025-08-01", "2025-08-31", max_cloud=30, catalog=cat)

    assert [s["id"] for s in scenes] == ["early", "late"]
    assert cat.kwargs["bbox"] == list(ingest.WILAYA_BBOXES["assaba"])
    assert cat.kwargs["datetime"] == "2025-08-01/2025-08-31"
    assert cat.kwargs["query"] == {"eo:cloud_cover": {"lt": 30}}
    assert cat.kwargs["collections"] == ["sentinel-2-l2a"]

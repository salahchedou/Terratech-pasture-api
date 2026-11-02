"""Offline tests for model.seasonal on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from model import seasonal as ss


def test_season_and_season_year():
    d = pd.Series(pd.to_datetime(["2025-07-01", "2025-10-31", "2025-11-01", "2026-06-30"]))
    assert ss.season_of(d).tolist() == ["rainy", "rainy", "dry", "dry"]
    assert ss.season_year(d).tolist() == [2025, 2025, 2025, 2025]


def _series(n_points=3, peak=0.3):
    rows = []
    dates = pd.date_range("2025-06-01", "2025-11-30", freq="10D")
    for p in range(n_points):
        for d in dates:
            doy = d.day_of_year
            savi = peak * np.exp(-((doy - 245) / 30) ** 2) + 0.05 + 0.01 * p
            rows.append({"point_id": p, "date": d, "SAVI": savi, "NDMI": savi - 0.3,
                         "NDVI": savi * 1.4, "lon": -11.4 + p, "lat": 16.6, "region": "assaba"})
    return pd.DataFrame(rows)


def test_rainy_season_features_only_use_july_to_october():
    feats = ss.rainy_season_features(_series())
    assert len(feats) == 3
    row = feats.iloc[0]
    assert row["season_year"] == 2025
    assert row["n_obs"] == 13  # 10-day steps falling in Jul-Oct
    assert 230 <= row["SAVI_peak_doy"] <= 260
    assert row["SAVI_late_slope"] < 0  # green-down in Sep-Oct
    assert row["SAVI_integral"] > 0
    assert row["region"] == "assaba" and row["lon"] == -11.4
    assert "NDVI_max" in feats and "NDVI_integral" not in feats


def test_points_with_too_few_observations_are_dropped():
    ts = _series(n_points=1)
    ts.loc[ts["date"].dt.month.isin([7, 8, 9, 10]), "SAVI"] = np.nan
    assert ss.rainy_season_features(ts).empty


def test_spatial_groups_keep_neighbours_together():
    g = ss.spatial_groups([-11.40, -11.401, -11.0], [16.60, 16.601, 16.60], block_km=20)
    assert g[0] == g[1] != g[2]


def test_regression_metrics_known_values():
    m = ss.regression_metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["r2"] == pytest.approx(0.5)


def test_spatial_cv_and_metrics_end_to_end():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "lon": rng.uniform(-13, -9, n),
        "lat": rng.uniform(15.5, 17.5, n),
        "SAVI_integral": rng.uniform(5, 30, n),
        "NDMI_end": rng.uniform(-0.3, 0.1, n),
    })
    df["biomass"] = 40 * df["SAVI_integral"] + 500 * df["NDMI_end"] + rng.normal(0, 30, n)
    df["region"] = np.where(df["lon"] < -11, "west", "east")
    df["season"] = np.where(rng.random(n) < 0.5, "dry_early", "dry_late")

    feats = ss.feature_columns(df)
    assert feats == ["SAVI_integral", "NDMI_end"]
    groups = ss.spatial_groups(df["lon"], df["lat"], block_km=40)
    df["pred"] = ss.cross_validate(df, feats, "biomass", groups, params={"n_estimators": 150})
    assert df["pred"].notna().all()

    table = ss.metrics_by(df, "biomass")
    overall = table.iloc[0]
    assert overall["region"] == "all" and overall["r2"] > 0.8
    assert set(table["region"]) == {"all", "west", "east"}
    assert set(table["season"]) == {"all", "dry_early", "dry_late"}


def test_cross_validate_refuses_too_few_groups():
    df = pd.DataFrame({"x": range(10), "y": range(10)})
    with pytest.raises(ValueError, match="spatial groups"):
        ss.cross_validate(df, ["x"], "y", [0] * 10, n_splits=5)


def test_temporal_split_never_trains_on_the_future():
    df = pd.DataFrame({"season_year": [2021, 2022, 2023, 2024]})
    train, test = ss.temporal_split(df, 2023)
    assert train["season_year"].max() < 2023 and test["season_year"].tolist() == [2023]

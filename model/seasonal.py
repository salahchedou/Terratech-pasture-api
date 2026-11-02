"""Seasonal AI predictor: forecast dry-season pasture quality from the rainy season.

The idea: how green and how moist the rangeland got during the rains
(July-October), and how fast it started to dry, tells us how much biomass
and crude protein will be left as the dry season (November-June) goes on.

Pipeline:
1. ``rainy_season_features``: turn per-point SAVI/NDMI time series into one
   row of season features per point and year (integrals, peak, late slope...).
2. ``spatial_groups``: assign points to square blocks of ~20 km so that
   neighbouring pixels never end up on both sides of a train/test split.
3. ``cross_validate``: GroupKFold over those blocks with LightGBM, giving
   honest out-of-fold predictions.
4. ``metrics_by``: RMSE, MAE and R2 per region and per season.
5. ``train`` / ``predict``: fit on all labelled data and forecast new points.

Labels (field biomass or a nutrition score) are not in the repo yet; every
function takes a pandas DataFrame so they can be plugged in when available.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

RAINY_MONTHS = (7, 8, 9, 10)
LATE_RAINY_MONTHS = (9, 10)  # green-down starts here
INDICES = ("SAVI", "NDMI")
COMPARISON_INDICES = ("NDVI",)  # kept only as a comparison feature
DEFAULT_BLOCK_KM = 20.0
MIN_OBS = 3  # need at least this many clear dates in the rainy season

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# Seasons
# ---------------------------------------------------------------------------


def season_of(dates: pd.Series) -> pd.Series:
    """'rainy' for July-October, 'dry' otherwise."""
    month = pd.to_datetime(dates).dt.month
    return pd.Series(np.where(month.isin(RAINY_MONTHS), "rainy", "dry"), index=dates.index)


def season_year(dates: pd.Series) -> pd.Series:
    """Year of the rainy season a date belongs to.

    July 2025 - June 2026 is season 2025: the 2025 rains feed the dry season
    that runs into 2026.
    """
    d = pd.to_datetime(dates)
    return (d.dt.year - (d.dt.month < 7).astype(int)).rename("season_year")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _trajectory_features(days: np.ndarray, values: np.ndarray, prefix: str) -> dict[str, float]:
    ok = np.isfinite(values)
    d, v = days[ok], values[ok]
    if d.size == 0:
        return {}
    order = np.argsort(d)
    d, v = d[order], v[order]
    feats = {
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_max": float(v.max()),
        f"{prefix}_min": float(v.min()),
        f"{prefix}_end": float(v[-1]),
        # Integral over the season (index x days): total "green-days".
        f"{prefix}_integral": float(np.trapezoid(v, d)) if d.size > 1 else 0.0,
        f"{prefix}_peak_doy": float(d[int(np.argmax(v))]),
    }
    return feats


def _late_slope(days: np.ndarray, values: np.ndarray) -> float:
    ok = np.isfinite(values)
    if ok.sum() < 2 or np.ptp(days[ok]) == 0:
        return np.nan
    return float(np.polyfit(days[ok], values[ok], 1)[0])  # index units per day


def rainy_season_features(
    ts: pd.DataFrame,
    *,
    point_col: str = "point_id",
    date_col: str = "date",
    indices: Sequence[str] = INDICES + COMPARISON_INDICES,
    min_obs: int = MIN_OBS,
) -> pd.DataFrame:
    """One feature row per (point, season_year) from a rainy-season time series.

    Args:
        ts: long table with one row per point and date, columns ``point_col``,
            ``date_col`` and the index columns (masked values as NaN). Extra
            static columns (lon, lat, region, terrain, rainfall...) are carried
            over using their first value per point.
        min_obs: points with fewer clear SAVI dates in the season are dropped.

    Returns:
        DataFrame indexed by nothing, with ``point_col``, ``season_year``,
        ``n_obs``, the static columns and features such as SAVI_integral,
        SAVI_max, SAVI_peak_doy, SAVI_late_slope, NDMI_end...
    """
    df = ts.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[df[date_col].dt.month.isin(RAINY_MONTHS)]
    df["season_year"] = season_year(df[date_col])
    df["_doy"] = df[date_col].dt.day_of_year.astype(float)
    present = [c for c in indices if c in df.columns]
    static = [c for c in df.columns if c not in {point_col, date_col, "season_year", "_doy", *present}]

    rows = []
    for (pid, year), g in df.groupby([point_col, "season_year"], sort=True):
        n_obs = int(np.isfinite(g["SAVI"]).sum()) if "SAVI" in g else 0
        if n_obs < min_obs:
            continue
        row: dict[str, Any] = {point_col: pid, "season_year": int(year), "n_obs": n_obs}
        for c in static:
            row[c] = g[c].iloc[0]
        days = g["_doy"].to_numpy()
        late = g[date_col].dt.month.isin(LATE_RAINY_MONTHS).to_numpy()
        for name in present:
            vals = g[name].to_numpy(dtype=float)
            if name in COMPARISON_INDICES:
                feats = _trajectory_features(days, vals, name)
                row.update({k: feats[k] for k in (f"{name}_max", f"{name}_mean") if k in feats})
                continue
            row.update(_trajectory_features(days, vals, name))
            row[f"{name}_late_slope"] = _late_slope(days[late], vals[late])
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame, extra: Sequence[str] = ()) -> list[str]:
    """Spectral feature columns produced by rainy_season_features, plus ``extra``."""
    prefixes = tuple(f"{n}_" for n in INDICES + COMPARISON_INDICES)
    cols = [c for c in df.columns if c.startswith(prefixes)]
    return cols + [c for c in extra if c in df.columns and c not in cols]


# ---------------------------------------------------------------------------
# Spatial validation
# ---------------------------------------------------------------------------


def spatial_groups(lon: Sequence[float], lat: Sequence[float], block_km: float = DEFAULT_BLOCK_KM) -> np.ndarray:
    """Integer block id per point from a ~``block_km`` square lattice.

    Used as ``groups`` in GroupKFold so whole blocks go to either train or test.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    dlat = block_km / 111.32
    dlon = block_km / (111.32 * np.cos(np.radians(np.nanmean(lat))))
    bx = np.floor(lon / dlon).astype(np.int64)
    by = np.floor(lat / dlat).astype(np.int64)
    _, ids = np.unique(np.stack([bx, by], axis=1), axis=0, return_inverse=True)
    return ids.ravel()


def _make_model(params: Mapping[str, Any] | None):
    from lightgbm import LGBMRegressor

    return LGBMRegressor(**{**DEFAULT_LGBM_PARAMS, **(params or {})})


def cross_validate(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str,
    groups: Sequence[int],
    *,
    n_splits: int = 5,
    params: Mapping[str, Any] | None = None,
) -> pd.Series:
    """Out-of-fold LightGBM predictions with GroupKFold over spatial blocks."""
    from sklearn.model_selection import GroupKFold

    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        raise ValueError(f"only {n_groups} spatial groups for {n_splits} folds; use smaller blocks")
    X, y = df[list(features)], df[target]
    oof = pd.Series(np.nan, index=df.index, name="pred")
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = _make_model(params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof.iloc[test_idx] = model.predict(X.iloc[test_idx])
    return oof


def temporal_split(df: pd.DataFrame, test_year: int, year_col: str = "season_year"):
    """Train on seasons before ``test_year``, test on ``test_year`` (no future leakage)."""
    return df[df[year_col] < test_year], df[df[year_col] == test_year]


# ---------------------------------------------------------------------------
# Metrics, training, inference
# ---------------------------------------------------------------------------


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "n": int(y_true.size),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": 1 - float((err ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan"),
    }


def metrics_by(
    df: pd.DataFrame,
    target: str,
    pred: str = "pred",
    by: Sequence[str] = ("region", "season"),
) -> pd.DataFrame:
    """RMSE / MAE / R2 for each value of each ``by`` column, plus an 'all' row.

    ``season`` here is the season of the label (e.g. early vs late dry season);
    columns missing from ``df`` are skipped.
    """
    by = [c for c in by if c in df.columns]
    rows = [{**{c: "all" for c in by}, **regression_metrics(df[target], df[pred])}]
    for c in by:
        for key, g in df.groupby(c):
            rows.append({**{k: "all" for k in by}, c: key, **regression_metrics(g[target], g[pred])})
    return pd.DataFrame(rows)


def train(df: pd.DataFrame, features: Sequence[str], target: str, params: Mapping[str, Any] | None = None):
    """Fit the final model on all labelled rows."""
    model = _make_model(params)
    model.fit(df[list(features)], df[target])
    return model


def predict(model, df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    return model.predict(df[list(features)])

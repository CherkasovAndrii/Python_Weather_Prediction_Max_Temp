"""Loading the weather dataset and turning it into a supervised learning problem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "DATE"
MONTH_COLUMN = "MONTH"
TARGET_SUFFIX = "_temp_max"
SEASONAL_FEATURES = ("doy_sin", "doy_cos")

# Physically plausible ranges in the units of the CSV. A few missing observations in the
# source data are encoded with sentinels (e.g. cloud cover -99, pressure 0), which would
# otherwise be fed to the model as real measurements.
VALID_RANGES: dict[str, tuple[float, float]] = {
    "_cloud_cover": (0, 9),  # oktas (9 = sky obscured)
    "_humidity": (0, 1),  # fraction
    "_pressure": (0.85, 1.1),  # 1000 hPa
    "_global_radiation": (0, 5),  # 100 W/m²
    "_precipitation": (0, 30),  # 10 mm
    "_sunshine": (0, 24),  # hours
    "_wind_speed": (0, 60),  # m/s
    "_wind_gust": (0, 80),  # m/s
    "_temp_mean": (-50, 50),  # °C
    "_temp_min": (-50, 50),
    "_temp_max": (-50, 50),
}


@dataclass(frozen=True)
class DatasetSplit:
    """Features and targets for the train / validation / test periods."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.y_train), "val": len(self.y_val), "test": len(self.y_test)}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV and return it with a parsed, sorted ``DatetimeIndex``.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file does not look like the weather prediction dataset.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{DATE_COLUMN}' column in {path}")

    df.index = pd.to_datetime(df.pop(DATE_COLUMN).astype(str), format="%Y%m%d")
    df.index.name = "date"
    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError("Dataset contains duplicate dates")
    return df


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace physically impossible values with the last valid observation.

    Forward-filling only uses the past, so no information from the future leaks into
    the features. Returns the cleaned frame and the number of fixed values per column.
    """
    out = df.copy()
    fixed: dict[str, int] = {}
    for column in out.columns:
        bounds = next((r for s, r in VALID_RANGES.items() if column.endswith(s)), None)
        if bounds is None:
            continue
        invalid = (out[column] < bounds[0]) | (out[column] > bounds[1])
        if invalid.any():
            fixed[column] = int(invalid.sum())
            out[column] = out[column].mask(invalid).ffill().bfill()
    return out, fixed


def available_cities(df: pd.DataFrame) -> list[str]:
    """Return every location that has a maximum-temperature column.

    Uses the column suffix rather than splitting on ``_`` so that multi-word
    locations such as ``DE_BILT`` are handled correctly.
    """
    return sorted(c.removesuffix(TARGET_SUFFIX) for c in df.columns if c.endswith(TARGET_SUFFIX))


def target_column(df: pd.DataFrame, city: str) -> str:
    """Return the name of the target column for ``city`` (case-insensitive)."""
    column = f"{city.upper()}{TARGET_SUFFIX}"
    if column not in df.columns:
        raise ValueError(
            f"Unknown city '{city}'. Available cities: {', '.join(available_cities(df))}"
        )
    return column


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the day of the year as a point on a circle (sin / cos).

    A cyclic encoding tells the model that 31 December and 1 January are
    neighbours, which a raw month number cannot do.
    """
    angle = 2 * np.pi * df.index.dayofyear / 365.25
    out = df.copy()
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    return out


def make_supervised(
    df: pd.DataFrame, city: str, horizon_days: int = 1
) -> tuple[pd.DataFrame, pd.Series]:
    """Build features ``X`` (observations on day *t*) and target ``y`` (max temp on *t + h*).

    The index of both outputs is the date of the observation, so a row always
    contains only information that was available when the forecast was made.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    target = target_column(df, city)
    features = add_seasonal_features(df.drop(columns=[MONTH_COLUMN], errors="ignore"))
    y = df[target].shift(-horizon_days).rename(f"{target}_t+{horizon_days}")

    X = features.iloc[:-horizon_days]
    y = y.iloc[:-horizon_days]
    return X, y


def chronological_split(
    X: pd.DataFrame, y: pd.Series, train_end: str, val_end: str
) -> DatasetSplit:
    """Split by date: train <= ``train_end`` < val <= ``val_end`` < test."""
    train_end_ts, val_end_ts = pd.Timestamp(train_end), pd.Timestamp(val_end)
    if train_end_ts >= val_end_ts:
        raise ValueError("train_end must be earlier than val_end")

    train = X.index <= train_end_ts
    val = (X.index > train_end_ts) & (X.index <= val_end_ts)
    test = X.index > val_end_ts

    if not (train.any() and val.any() and test.any()):
        raise ValueError("Every split must contain at least one sample; check the dates")

    return DatasetSplit(X[train], y[train], X[val], y[val], X[test], y[test])

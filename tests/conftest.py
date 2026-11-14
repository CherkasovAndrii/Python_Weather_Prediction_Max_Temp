from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REAL_DATA = Path(__file__).resolve().parents[1] / "data" / "weather_prediction_dataset.csv"


def _synthetic_frame(
    start: str = "2000-01-01", end: str = "2010-01-01", seed: int = 0
) -> pd.DataFrame:
    """Small dataset with the same schema as the real one (incl. a two-word city)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    season = np.sin(2 * np.pi * dates.dayofyear / 365.25)
    return pd.DataFrame(
        {
            "DATE": dates.strftime("%Y%m%d").astype(int),
            "MONTH": dates.month,
            "ROMA_humidity": rng.uniform(0.4, 1.0, len(dates)),
            "ROMA_temp_max": 20 + 8 * season + rng.normal(0, 2, len(dates)),
            "DE_BILT_temp_max": 14 + 9 * season + rng.normal(0, 2, len(dates)),
            "OSLO_pressure": rng.normal(1.01, 0.01, len(dates)),
            "OSLO_temp_max": 10 + 11 * season + rng.normal(0, 2, len(dates)),
        }
    )


@pytest.fixture
def synthetic_csv(tmp_path: Path) -> Path:
    path = tmp_path / "weather.csv"
    _synthetic_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def real_csv() -> Path:
    if not REAL_DATA.is_file():
        pytest.skip("real dataset not available")
    return REAL_DATA

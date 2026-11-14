import numpy as np
import pandas as pd
import pytest

from maxtemp.baselines import PersistenceForecast, build_ridge
from maxtemp.metrics import interval_coverage, regression_metrics


class TestPersistence:
    def test_predicts_todays_value(self):
        X = pd.DataFrame({"ROMA_temp_max": [10.0, 12.5, 9.0], "other": [1, 2, 3]})
        model = PersistenceForecast("ROMA_temp_max").fit(X)
        np.testing.assert_array_equal(model.predict(X), [10.0, 12.5, 9.0])

    def test_missing_column_raises(self):
        with pytest.raises(KeyError):
            PersistenceForecast("ROMA_temp_max").fit(pd.DataFrame({"x": [1.0]}))


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = 2 * X["a"] - 3 * X["b"] + 5
    model = build_ridge().fit(X, y)
    assert regression_metrics(y, model.predict(X))["r2"] > 0.999


class TestRegressionMetrics:
    def test_perfect_forecast(self):
        m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert m == {"rmse": 0.0, "mae": 0.0, "r2": 1.0}

    def test_known_values(self):
        m = regression_metrics([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 3.0, -3.0])
        assert m["mae"] == pytest.approx(2.0)
        assert m["rmse"] == pytest.approx(np.sqrt(5.0))

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            regression_metrics([1.0, 2.0], [1.0])


class TestIntervalCoverage:
    def test_all_inside(self):
        assert interval_coverage([1.0, 2.0], [1.0, 2.0], [0.1, 0.1]) == 1.0

    def test_half_inside(self):
        assert interval_coverage([0.0, 10.0], [0.0, 0.0], [1.0, 1.0]) == 0.5

    def test_gaussian_calibration(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1, 100_000)
        cov = interval_coverage(y, np.zeros_like(y), np.ones_like(y))
        assert cov == pytest.approx(0.95, abs=0.005)

import numpy as np
import pandas as pd
import pytest

from maxtemp.data import (
    available_cities,
    chronological_split,
    clean_dataset,
    load_dataset,
    make_supervised,
    target_column,
)


@pytest.fixture
def df(synthetic_csv):
    return load_dataset(synthetic_csv)


class TestLoadDataset:
    def test_parses_dates_into_sorted_index(self, df):
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert "DATE" not in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.csv")

    def test_wrong_schema_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="DATE"):
            load_dataset(path)

    def test_duplicate_dates_raise(self, tmp_path):
        path = tmp_path / "dup.csv"
        pd.DataFrame({"DATE": [20000101, 20000101], "X_temp_max": [1.0, 2.0]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_dataset(path)


class TestCleanDataset:
    def test_sentinels_replaced_by_previous_valid_value(self, df):
        dirty = df.copy()
        dirty.iloc[10, dirty.columns.get_loc("OSLO_pressure")] = 0.0
        dirty.iloc[20, dirty.columns.get_loc("ROMA_humidity")] = -99
        clean, fixed = clean_dataset(dirty)
        assert fixed == {"ROMA_humidity": 1, "OSLO_pressure": 1}
        assert clean["OSLO_pressure"].iloc[10] == dirty["OSLO_pressure"].iloc[9]
        assert clean["ROMA_humidity"].iloc[20] == dirty["ROMA_humidity"].iloc[19]

    def test_leading_invalid_value_is_backfilled(self, df):
        dirty = df.copy()
        dirty.iloc[0, dirty.columns.get_loc("OSLO_pressure")] = 0.0
        clean, _ = clean_dataset(dirty)
        assert clean["OSLO_pressure"].iloc[0] == dirty["OSLO_pressure"].iloc[1]

    def test_clean_data_is_unchanged(self, df):
        clean, fixed = clean_dataset(df)
        assert fixed == {}
        pd.testing.assert_frame_equal(clean, df)

    def test_input_is_not_mutated(self, df):
        dirty = df.copy()
        dirty.iloc[5, dirty.columns.get_loc("OSLO_pressure")] = 0.0
        snapshot = dirty.copy()
        clean_dataset(dirty)
        pd.testing.assert_frame_equal(dirty, snapshot)


class TestCities:
    def test_multi_word_city_is_not_truncated(self, df):
        # The original notebooks split on "_" and reported "DE" instead of "DE_BILT".
        assert available_cities(df) == ["DE_BILT", "OSLO", "ROMA"]

    def test_target_column_is_case_insensitive(self, df):
        assert target_column(df, "roma") == "ROMA_temp_max"

    def test_unknown_city_lists_alternatives(self, df):
        with pytest.raises(ValueError, match="Available cities: DE_BILT, OSLO, ROMA"):
            target_column(df, "PARIS")


class TestMakeSupervised:
    def test_target_is_next_day_value(self, df):
        X, y = make_supervised(df, "ROMA", horizon_days=1)
        np.testing.assert_allclose(y.to_numpy(), df["ROMA_temp_max"].to_numpy()[1:])
        np.testing.assert_allclose(
            X["ROMA_temp_max"].to_numpy(), df["ROMA_temp_max"].to_numpy()[:-1]
        )

    @pytest.mark.parametrize("horizon", [1, 3, 7])
    def test_lengths_and_alignment(self, df, horizon):
        X, y = make_supervised(df, "OSLO", horizon_days=horizon)
        assert len(X) == len(y) == len(df) - horizon
        assert X.index.equals(y.index)
        assert y.iloc[0] == df["OSLO_temp_max"].iloc[horizon]

    def test_no_missing_targets(self, df):
        _, y = make_supervised(df, "ROMA")
        assert not y.isna().any()

    def test_month_dropped_and_seasonality_added(self, df):
        X, _ = make_supervised(df, "ROMA")
        assert "MONTH" not in X.columns
        assert {"doy_sin", "doy_cos"} <= set(X.columns)
        radius = X["doy_sin"] ** 2 + X["doy_cos"] ** 2
        np.testing.assert_allclose(radius, 1.0)

    def test_invalid_horizon(self, df):
        with pytest.raises(ValueError):
            make_supervised(df, "ROMA", horizon_days=0)


class TestChronologicalSplit:
    def test_periods_do_not_overlap_and_are_ordered(self, df):
        X, y = make_supervised(df, "ROMA")
        s = chronological_split(X, y, "2006-12-31", "2007-12-31")
        assert s.X_train.index.max() < s.X_val.index.min()
        assert s.X_val.index.max() < s.X_test.index.min()
        assert sum(s.sizes().values()) == len(X)

    def test_val_is_exactly_one_year(self, df):
        X, y = make_supervised(df, "ROMA")
        s = chronological_split(X, y, "2006-12-31", "2007-12-31")
        assert s.sizes()["val"] == 365

    def test_reversed_dates_raise(self, df):
        X, y = make_supervised(df, "ROMA")
        with pytest.raises(ValueError, match="earlier"):
            chronological_split(X, y, "2008-01-01", "2007-01-01")

    def test_empty_split_raises(self, df):
        X, y = make_supervised(df, "ROMA")
        with pytest.raises(ValueError, match="at least one sample"):
            chronological_split(X, y, "2015-01-01", "2016-01-01")


class TestRealDataset:
    """Data-quality checks on the committed CSV."""

    def test_shape_and_no_missing_values(self, real_csv):
        df = load_dataset(real_csv)
        assert df.shape == (3654, 164)
        assert not df.isna().any().any()

    def test_daily_without_gaps(self, real_csv):
        df = load_dataset(real_csv)
        assert (df.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()

    def test_known_sentinel_values_are_cleaned(self, real_csv):
        _, fixed = clean_dataset(load_dataset(real_csv))
        # Documented in the README: 35 sentinel values in four columns.
        assert fixed == {
            "STOCKHOLM_cloud_cover": 2,
            "STOCKHOLM_pressure": 3,
            "STOCKHOLM_sunshine": 29,
            "TOURS_pressure": 1,
        }

    def test_eighteen_locations_with_plausible_temperatures(self, real_csv):
        df = load_dataset(real_csv)
        cities = available_cities(df)
        assert len(cities) == 18
        temps = df[[f"{c}_temp_max" for c in cities]]
        assert temps.min().min() > -40
        assert temps.max().max() < 50

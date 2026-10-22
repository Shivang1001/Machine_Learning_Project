import pandas as pd
import pytest

from src.m4b_demand import forecast_demand

COLS = ["week_start", "forecast", "lower", "upper", "techs_needed"]


def test_forecast_demand_structure():
    df = forecast_demand(branch_id="NCR", weeks=4)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == COLS
    assert len(df) == 4
    assert all(isinstance(x, int) for x in df["techs_needed"])


def test_forecast_starts_2025_12_29_and_is_weekly():
    df = forecast_demand("PUN", 4)
    assert df["week_start"].iloc[0] == pd.Timestamp("2025-12-29")
    assert (df["week_start"].diff().dropna() == pd.Timedelta(days=7)).all()


def test_interval_contains_forecast_and_widens():
    df = forecast_demand("JAI", 4)
    assert ((df["lower"] <= df["forecast"]) & (df["forecast"] <= df["upper"])).all()
    width = (df["upper"] - df["lower"]) / df["forecast"]
    assert width.is_monotonic_increasing


def test_deterministic():
    pd.testing.assert_frame_equal(forecast_demand("NCR", 4), forecast_demand("NCR", 4))


def test_forecast_demand_invalid_branch():
    with pytest.raises(ValueError):
        forecast_demand(branch_id="INVALID", weeks=4)


def test_forecast_demand_invalid_weeks():
    with pytest.raises(ValueError):
        forecast_demand(branch_id="NCR", weeks=0)
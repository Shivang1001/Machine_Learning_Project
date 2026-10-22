import numpy as np
import pytest

from src import config
from src import m3b_quote as m

ARGS = dict(fault_category="geyser", appliance_type="geyser", premises_type="house",
            house_age_years=20, urgency="high", branch_id="NCR", experience_years=5,
            skill_level=2, month=12, source_channel="website", response_time_min=30,
            hour=10, day_of_week=2)


class FakeModel:
    def predict_proba(self, X):
        p = 0.9 / (1 + np.exp(6 * (X["quote_ratio"].to_numpy() - 1.08)))
        return np.column_stack([1 - p, p])


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    monkeypatch.setattr(m, "_lead_model", lambda: FakeModel())
    monkeypatch.setattr(m, "_rate_table", lambda: {(f, p): 1500.0 for f in config.FAULTS for p in config.PREMISES})
    monkeypatch.setattr(m, "_predict_cost", lambda **k: 1000.0)


def test_keys_and_types():
    r = m.recommend_quote(**ARGS)
    assert set(r) == {"price", "win_prob", "expected_profit", "cost_inr", "typical_price", "curve"}
    assert all(isinstance(r[k], float) for k in r if k != "curve")
    assert set(r["curve"][0]) == {"price", "win_prob", "expected_profit"}


def test_optimum_and_range():
    r = m.recommend_quote(**ARGS)
    assert r["typical_price"] == pytest.approx(1500 * 1.12)
    assert 1000 <= r["price"] <= 2600
    assert r["expected_profit"] == pytest.approx(r["win_prob"] * (r["price"] - r["cost_inr"]))
    assert r["expected_profit"] == pytest.approx(max(c["expected_profit"] for c in r["curve"]))
    prices = [c["price"] for c in r["curve"]]
    assert prices == sorted(prices) and prices[1] - prices[0] == pytest.approx(5.0)


@pytest.mark.parametrize("field,bad", [("fault_category", "x"), ("appliance_type", "x"),
                                       ("premises_type", "x"), ("urgency", "x"),
                                       ("branch_id", "x"), ("source_channel", "x")])
def test_invalid_enum_raises(field, bad):
    with pytest.raises(ValueError):
        m.recommend_quote(**{**ARGS, field: bad})
        
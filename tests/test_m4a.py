from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src import m4a_segments as m
from src.train_m4a import make_pipeline

FIX = Path("tests/fixtures")
NAMES = ["loyal_high_value", "at_risk", "new_one_time", "regular_seasonal"]


def _fixtures():
    """200 jobs and 60 customers (only the columns M4A uses); saved to tests/fixtures."""
    rng = np.random.default_rng(42)
    ids = [f"C-{i:05d}" for i in range(1, 61)]
    cust = pd.DataFrame({"customer_id": ids, "branch_id": rng.choice(config.BRANCHES, 60),
                         "premises_type": rng.choice(config.PREMISES, 60),
                         "house_age_years": rng.integers(0, 51, 60).astype(float), "signup_date": "2023-12-01"})
    jobs = pd.DataFrame({"job_id": [f"J-{i:06d}" for i in range(1, 201)],
                         "customer_id": rng.choice(ids, 200),
                         "job_date": pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 731, 200), unit="D"),
                         "fault_category": rng.choice(config.FAULTS, 200),
                         "final_price_inr": rng.integers(800, 5000, 200).astype(float)})
    FIX.mkdir(parents=True, exist_ok=True)
    if not (FIX / "m4a_jobs.csv").exists():
        jobs.to_csv(FIX / "m4a_jobs.csv", index=False); cust.to_csv(FIX / "m4a_customers.csv", index=False)
    return cust, jobs


@pytest.fixture
def ready(monkeypatch):
    cust, jobs = _fixtures()
    monkeypatch.setattr(m, "load", lambda name: {"jobs": jobs, "customers": cust}[name])
    bundle = m.fit_segments(m.snapshot("2025-06-30"))
    tr = m.snapshot("2024-12-31"); tr["segment"] = m.assign_segments(tr, bundle)
    pipe = make_pipeline("logreg").fit(tr[m.REPEAT_FEATURES], np.arange(len(tr)) % 2)  # interface test only
    monkeypatch.setattr(m, "_kmeans", lambda: bundle)
    monkeypatch.setattr(m, "_repeat", lambda: pipe)
    return bundle


def test_get_segments_shape(ready):
    df = m.get_segments("2025-06-30")
    assert list(df.columns) == ["customer_id", "recency_days", "frequency", "monetary", "segment", "repeat_prob_6m"]
    assert set(df["segment"]) <= set(NAMES)
    assert df["repeat_prob_6m"].between(0, 1).all()
    assert df["customer_id"].is_unique and df["customer_id"].is_monotonic_increasing


def test_name_map_covers_all_names(ready):
    assert sorted(ready["name_map"].values()) == sorted(NAMES)


def test_rfm_uses_only_past_jobs(ready):
    r = m.rfm("2024-06-30")
    assert (r["recency_days"] >= 0).all() and r["frequency"].min() >= 1


def test_deterministic(ready):
    pd.testing.assert_frame_equal(m.get_segments(), m.get_segments())


@pytest.mark.parametrize("bad", ["not-a-date", "2030-01-01", "2020-01-01"])
def test_invalid_as_of_raises(ready, bad):
    with pytest.raises(ValueError):
        m.get_segments(bad)
"""Contract unit tests for Module M2."""

import pytest
from src.m2_estimator import predict_job


def test_predict_job_signature_and_types():
    res = predict_job(
        fault_category="ac",
        appliance_type="ac",
        premises_type="apartment",
        house_age_years=15,
        urgency="high",
        branch_id="NCR",
        experience_years=5,
        skill_level=2,
        month=5,
    )
    assert isinstance(res, dict)
    assert set(res.keys()) == {"duration_hours", "duration_low", "duration_high", "cost_inr"}
    for k, v in res.items():
        assert isinstance(v, float), f"{k} is not a float"

    assert res["duration_low"] <= res["duration_hours"] <= res["duration_high"]
    assert res["cost_inr"] > 199.0


def test_predict_job_invalid_enums():
    with pytest.raises(ValueError, match="Invalid fault_category"):
        predict_job("plumbing", "ac", "apartment", 15, "high", "NCR", 5, 2, 5)

    with pytest.raises(ValueError, match="Invalid branch_id"):
        predict_job("ac", "ac", "apartment", 15, "high", "MUM", 5, 2, 5)
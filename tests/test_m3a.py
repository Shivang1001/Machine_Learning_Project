import pytest
from src.m3a_leads import score_lead

def test_score_lead_runs_and_returns_float():
    prob = score_lead(
        source_channel="google_ads",
        response_time_min=30,
        hour=14,
        day_of_week=2,
        appliance_type="ac",
        premises_type="apartment",
        quote_ratio=1.1,
        urgency="medium",
        branch_id="NCR"
    )
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

def test_score_lead_invalid_enum_raises_value_error():
    with pytest.raises(ValueError):
        score_lead(
            source_channel="invalid_channel",
            response_time_min=30,
            hour=14,
            day_of_week=2,
            appliance_type="ac",
            premises_type="apartment",
            quote_ratio=1.1,
            urgency="medium",
            branch_id="NCR"
        )
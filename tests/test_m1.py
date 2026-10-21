"""Contract unit tests for Module M1."""

import pytest
from src.config import FAULTS, PARTS, URGENCY
from src.m1_triage import predict_triage


def test_predict_triage_signature_and_keys():
    res = predict_triage(
        text="Main DB switch tripping repeatedly and switchboard sparked",
        appliance_type="switchboard",
        premises_type="apartment",
        house_age_years=12,
        hour=14
    )
    assert isinstance(res, dict)
    assert set(res.keys()) == {"fault", "fault_probs", "urgency", "urgency_probs", "parts"}

    # Fault checks
    assert res["fault"] in FAULTS
    assert set(res["fault_probs"].keys()) == set(FAULTS)
    assert abs(sum(res["fault_probs"].values()) - 1.0) < 0.01

    # Urgency checks
    assert res["urgency"] in URGENCY
    assert set(res["urgency_probs"].keys()) == set(URGENCY)
    assert abs(sum(res["urgency_probs"].values()) - 1.0) < 0.01

    # Parts checks
    assert len(res["parts"]) == 6
    for item in res["parts"]:
        assert item["part"] in PARTS
        assert 0.0 <= item["prob"] <= 1.0

    # Probabilities sorted descending
    probs = [x["prob"] for x in res["parts"]]
    assert probs == sorted(probs, reverse=True)


def test_predict_triage_invalid_enums():
    with pytest.raises(ValueError, match="Invalid appliance_type"):
        predict_triage("wiring issue", appliance_type="invalid_appliance")

    with pytest.raises(ValueError, match="Invalid premises_type"):
        predict_triage("wiring issue", premises_type="mansion")

    with pytest.raises(ValueError, match="Invalid hour"):
        predict_triage("wiring issue", hour=25)
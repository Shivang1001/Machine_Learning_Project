"""Contract unit tests for Module M5 (Impact Evaluation and App Readiness)."""

import json
from pathlib import Path
from src.evaluate_impact import evaluate_all

ROOT_DIR = Path(__file__).resolve().parent.parent


def test_evaluate_impact_writes_valid_contract_json():
    res = evaluate_all()
    m5_path = ROOT_DIR / "reports" / "metrics" / "m5.json"
    assert m5_path.exists()

    loaded = json.loads(m5_path.read_text(encoding="utf-8"))
    assert loaded["module"] == "M5"
    assert loaded["contract_version"] == "1.0"
    assert loaded["random_state"] == 42
    assert isinstance(loaded["assumptions"], list) and len(loaded["assumptions"]) >= 4

    imp = loaded["impact_summary"]
    assert imp["repeat_rate_ml"] <= imp["repeat_rate_baseline"]
    assert imp["profit_per_lead_ml"] > imp["profit_per_lead_baseline"]
    assert imp["top30_conv_ml"] > imp["top30_conv_random"]


def test_streamlit_app_file_exists_and_has_exact_tabs():
    app_path = ROOT_DIR / "app" / "streamlit_app.py"
    assert app_path.exists()
    content = app_path.read_text(encoding="utf-8")
    for required_tab in ["Triage & Quote", "Demand", "Customers", "Models & Impact"]:
        assert required_tab in content
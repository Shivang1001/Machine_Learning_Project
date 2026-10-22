from pathlib import Path
from typing import Dict
import joblib
import numpy as np
import pandas as pd
from src.config import APPLIANCES, BRANCHES, FAULTS, PREMISES, URGENCY

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_DUR_MODEL = None
_DUR_LOW_MODEL = None
_DUR_HIGH_MODEL = None
_COST_MODEL = None


def _get_models():
    """Lazily loads all four fitted M2 regression pipelines from disk."""
    global _DUR_MODEL, _DUR_LOW_MODEL, _DUR_HIGH_MODEL, _COST_MODEL
    if _DUR_MODEL is None:
        for name in ["m2_duration.joblib", "m2_duration_low.joblib", "m2_duration_high.joblib", "m2_cost.joblib"]:
            p = MODEL_DIR / name
            if not p.exists():
                raise FileNotFoundError(f"Missing model file {p}. Run 'python -m src.train_m2' first.")
        _DUR_MODEL = joblib.load(MODEL_DIR / "m2_duration.joblib")
        _DUR_LOW_MODEL = joblib.load(MODEL_DIR / "m2_duration_low.joblib")
        _DUR_HIGH_MODEL = joblib.load(MODEL_DIR / "m2_duration_high.joblib")
        _COST_MODEL = joblib.load(MODEL_DIR / "m2_cost.joblib")
    return _DUR_MODEL, _DUR_LOW_MODEL, _DUR_HIGH_MODEL, _COST_MODEL


def predict_job(
    fault_category: str,
    appliance_type: str,
    premises_type: str,
    house_age_years: int,
    urgency: str,
    branch_id: str,
    experience_years: int,
    skill_level: int,
    month: int,
) -> Dict[str, float]:
    """Predicts job duration (with P10-P90 interval) and business cost in INR."""
    if fault_category not in FAULTS:
        raise ValueError(f"Invalid fault_category '{fault_category}'. Expected one of {FAULTS}")
    if appliance_type not in APPLIANCES:
        raise ValueError(f"Invalid appliance_type '{appliance_type}'. Expected one of {APPLIANCES}")
    if premises_type not in PREMISES:
        raise ValueError(f"Invalid premises_type '{premises_type}'. Expected one of {PREMISES}")
    if urgency not in URGENCY:
        raise ValueError(f"Invalid urgency '{urgency}'. Expected one of {URGENCY}")
    if branch_id not in BRANCHES:
        raise ValueError(f"Invalid branch_id '{branch_id}'. Expected one of {BRANCHES}")
    if not (1 <= int(skill_level) <= 3):
        raise ValueError(f"Invalid skill_level '{skill_level}'. Expected 1, 2, or 3")
    if not (1 <= int(month) <= 12):
        raise ValueError(f"Invalid month '{month}'. Expected integer 1..12")

    dur_model, dur_low_model, dur_high_model, cost_model = _get_models()

    X = pd.DataFrame([{
        "fault_category": fault_category,
        "appliance_type": appliance_type,
        "premises_type": premises_type,
        "house_age_years": float(house_age_years) if house_age_years is not None else np.nan,
        "urgency": urgency,
        "branch_id": branch_id,
        "experience_years": float(experience_years),
        "skill_level": float(skill_level),
        "month": int(month),
    }])

    pred_dur = float(dur_model.predict(X)[0])
    pred_low = float(dur_low_model.predict(X)[0])
    pred_high = float(dur_high_model.predict(X)[0])

    # Ensure physical ordering of intervals
    pred_dur = max(0.2, pred_dur)
    pred_low = max(0.1, min(pred_low, pred_dur))
    pred_high = max(pred_dur, pred_high)

    # Cost model is trained on log1p(job_cost_inr), back-transform via expm1
    pred_log_cost = float(cost_model.predict(X)[0])
    pred_cost = float(np.expm1(pred_log_cost))

    return {
        "duration_hours": round(pred_dur, 4),
        "duration_low": round(pred_low, 4),
        "duration_high": round(pred_high, 4),
        "cost_inr": round(pred_cost, 2),
    }
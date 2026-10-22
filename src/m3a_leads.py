import joblib
import pandas as pd
from pathlib import Path
from src.config import CHANNELS, APPLIANCES, PREMISES, URGENCY, BRANCHES

_MODEL = None

def _load_model():
    global _MODEL
    if _MODEL is None:
        model_path = Path("models/m3a_lead.joblib")
        if not model_path.exists():
            raise FileNotFoundError("Model file not found. Run python -m src.train_m3a first.")
        _MODEL = joblib.load(model_path)
    return _MODEL

def score_lead(
    source_channel: str,
    response_time_min: float,
    hour: int,
    day_of_week: int,
    appliance_type: str,
    premises_type: str,
    quote_ratio: float,
    urgency: str,
    branch_id: str
) -> float:
    """Score a lead and return the calibrated probability of booking."""
    if source_channel not in CHANNELS:
        raise ValueError(f"Invalid source_channel: {source_channel}")
    if appliance_type not in APPLIANCES:
        raise ValueError(f"Invalid appliance_type: {appliance_type}")
    if premises_type not in PREMISES:
        raise ValueError(f"Invalid premises_type: {premises_type}")
    if urgency not in URGENCY:
        raise ValueError(f"Invalid urgency: {urgency}")
    if branch_id not in BRANCHES:
        raise ValueError(f"Invalid branch_id: {branch_id}")

    model = _load_model()
    
    row = pd.DataFrame([{
        "source_channel": source_channel,
        "response_time_min": response_time_min,
        "hour": hour,
        "day_of_week": day_of_week,
        "appliance_type": appliance_type,
        "premises_type": premises_type,
        "quote_ratio": quote_ratio,
        "urgency": urgency,
        "branch_id": branch_id
    }])
    
    prob = model.predict_proba(row)[0, 1]
    return float(prob)
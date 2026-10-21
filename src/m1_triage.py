"""M1: Real-time job triage inference for fault classification, urgency scoring, and parts prediction."""

import re
import string
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
from src.config import APPLIANCES, FAULTS, PARTS, PREMISES, URGENCY

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

_FAULT_MODEL = None
_URGENCY_MODEL = None
_PARTS_BUNDLE = None


def clean_text(text: str) -> str:
    """Lowercases text and strips all standard ASCII punctuation."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))


def _get_models():
    """Lazily loads fitted pipeline artifacts from disk."""
    global _FAULT_MODEL, _URGENCY_MODEL, _PARTS_BUNDLE
    if _FAULT_MODEL is None:
        fault_path = MODEL_DIR / "m1_fault.joblib"
        if not fault_path.exists():
            raise FileNotFoundError(f"Model file not found: {fault_path}. Run 'python -m src.train_m1' first.")
        _FAULT_MODEL = joblib.load(fault_path)

    if _URGENCY_MODEL is None:
        urgency_path = MODEL_DIR / "m1_urgency.joblib"
        if not urgency_path.exists():
            raise FileNotFoundError(f"Model file not found: {urgency_path}. Run 'python -m src.train_m1' first.")
        _URGENCY_MODEL = joblib.load(urgency_path)

    if _PARTS_BUNDLE is None:
        parts_path = MODEL_DIR / "m1_parts.joblib"
        if not parts_path.exists():
            raise FileNotFoundError(f"Model file not found: {parts_path}. Run 'python -m src.train_m1' first.")
        _PARTS_BUNDLE = joblib.load(parts_path)

    return _FAULT_MODEL, _URGENCY_MODEL, _PARTS_BUNDLE


def predict_triage(
    text: str,
    appliance_type: str = "unknown",
    premises_type: str = "apartment",
    house_age_years: int = 15,
    hour: int = 12
) -> Dict[str, Any]:
    """Infers fault category, urgency level, and top 6 required spare parts from an inquiry."""
    if appliance_type not in APPLIANCES:
        raise ValueError(f"Invalid appliance_type '{appliance_type}'. Expected one of {APPLIANCES}")
    if premises_type not in PREMISES:
        raise ValueError(f"Invalid premises_type '{premises_type}'. Expected one of {PREMISES}")
    if not (0 <= hour <= 23):
        raise ValueError(f"Invalid hour '{hour}'. Expected integer between 0 and 23")

    fault_model, urgency_model, parts_bundle = _get_models()

    cleaned = clean_text(text)
    input_df = pd.DataFrame([{
        "complaint_text": cleaned,
        "appliance_type": appliance_type,
        "premises_type": premises_type,
        "house_age_years": float(house_age_years) if house_age_years is not None else np.nan,
        "created_hour": int(hour)
    }])

    # 1. Fault classification
    fault_probs_arr = fault_model.predict_proba(input_df)[0]
    fault_classes = list(fault_model.classes_)
    fault_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(fault_classes, fault_probs_arr)}

    # Normalize sum to 1.0
    tot_f = sum(fault_probs.values())
    fault_probs = {k: round(v / tot_f, 4) for k, v in fault_probs.items()}
    pred_fault = max(fault_probs, key=fault_probs.get)

    # 2. Urgency classification
    urg_probs_arr = urgency_model.predict_proba(input_df)[0]
    urg_classes = list(urgency_model.classes_)
    urgency_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(urg_classes, urg_probs_arr)}
    tot_u = sum(urgency_probs.values())
    urgency_probs = {k: round(v / tot_u, 4) for k, v in urgency_probs.items()}
    pred_urgency = max(urgency_probs, key=urgency_probs.get)

    # 3. Parts multi-label prediction (using predicted fault at inference)
    parts_input_df = input_df.copy()
    parts_input_df["predicted_fault"] = pred_fault

    parts_model = parts_bundle["pipeline"]
    parts_labels = parts_bundle["parts"]

    raw_part_probs = parts_model.predict_proba(parts_input_df)[0]
    parts_list = []
    for p_name, p_prob in zip(parts_labels, raw_part_probs):
        parts_list.append({"part": p_name, "prob": round(float(p_prob), 4)})

    parts_list.sort(key=lambda x: x["prob"], reverse=True)
    top_6_parts = parts_list[:6]

    return {
        "fault": pred_fault,
        "fault_probs": fault_probs,
        "urgency": pred_urgency,
        "urgency_probs": urgency_probs,
        "parts": top_6_parts
    }
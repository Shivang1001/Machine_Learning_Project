"""M3B quote optimiser: choose the price that maximises expected profit."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.io import load, split

LEAD_MODEL_PATH = Path("models/m3a_lead.joblib")
METRICS_PATH = Path("reports/metrics/m3b.json")
LOW_MULT, HIGH_MULT, STEP_INR, BASELINE_MULT = 1.00, 2.60, 5.0, 1.30
LEAD_COLS = ["source_channel", "response_time_min", "hour", "day_of_week",
             "appliance_type", "premises_type", "quote_ratio", "urgency", "branch_id"]


def _check_enum(name: str, value: object, allowed: list) -> None:
    """Raise ValueError if value is not in the allowed list."""
    if value not in allowed:
        raise ValueError(f"{name}={value!r} is not valid; allowed: {allowed}")


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    """Raise ValueError if value is outside [lo, hi]."""
    if not (lo <= value <= hi):
        raise ValueError(f"{name}={value!r} must be between {lo} and {hi}")


@lru_cache(maxsize=1)
def _lead_model():
    """Load the calibrated M3A model once."""
    return joblib.load(LEAD_MODEL_PATH)


@lru_cache(maxsize=1)
def _rate_table() -> dict:
    """Load rate_card as {(fault, premises): base_price_inr}."""
    df = load("rate_card")
    return {(r.fault_category, r.premises_type): float(r.base_price_inr) for r in df.itertuples()}


def _typical_price(fault: str, premises: str, urgency: str) -> float:
    """Rate-card price, plus 12% when urgency is high."""
    base = _rate_table().get((fault, premises))
    if base is None:
        raise ValueError(f"No rate card entry for fault={fault!r}, premises={premises!r}")
    return base * (1.12 if urgency == "high" else 1.0)


def _predict_cost(**kw) -> float:
    """Predicted job cost in INR from the M2 interface."""
    from src.m2_estimator import predict_job
    return float(predict_job(**kw)["cost_inr"])


def _win_probs(prices: np.ndarray, typical: float, ctx: dict) -> np.ndarray:
    """Calibrated M3A booking probability for each candidate price."""
    n = len(prices)
    X = pd.DataFrame({c: [ctx[c]] * n for c in LEAD_COLS if c != "quote_ratio"})
    X["quote_ratio"] = prices / typical
    return _lead_model().predict_proba(X[LEAD_COLS])[:, 1]


def _optimise(cost: float, typical: float, ctx: dict):
    """Return prices, win probs, expected profits and the best index."""
    prices = np.arange(cost * LOW_MULT, cost * HIGH_MULT + 1e-9, STEP_INR)
    wp = _win_probs(prices, typical, ctx)
    profit = wp * (prices - cost)
    return prices, wp, profit, int(np.argmax(profit))  # argmax = first max = lowest price


def recommend_quote(fault_category: str, appliance_type: str, premises_type: str,
                    house_age_years: int, urgency: str, branch_id: str,
                    experience_years: int, skill_level: int, month: int,
                    source_channel: str, response_time_min: int, hour: int,
                    day_of_week: int) -> dict:
    """Recommend the profit-maximising quote for one lead."""
    _check_enum("fault_category", fault_category, config.FAULTS)
    _check_enum("appliance_type", appliance_type, config.APPLIANCES)
    _check_enum("premises_type", premises_type, config.PREMISES)
    _check_enum("urgency", urgency, config.URGENCY)
    _check_enum("branch_id", branch_id, config.BRANCHES)
    _check_enum("source_channel", source_channel, config.CHANNELS)
    _check_range("skill_level", skill_level, 1, 3)
    _check_range("month", month, 1, 12)
    _check_range("hour", hour, 0, 23)
    _check_range("day_of_week", day_of_week, 0, 6)
    cost = _predict_cost(fault_category=fault_category, appliance_type=appliance_type,
                         premises_type=premises_type, house_age_years=house_age_years,
                         urgency=urgency, branch_id=branch_id, experience_years=experience_years,
                         skill_level=skill_level, month=month)
    typical = _typical_price(fault_category, premises_type, urgency)
    ctx = {"source_channel": source_channel, "response_time_min": response_time_min,
           "hour": hour, "day_of_week": day_of_week, "appliance_type": appliance_type,
           "premises_type": premises_type, "urgency": urgency, "branch_id": branch_id}
    prices, wp, profit, best = _optimise(cost, typical, ctx)
    curve = [{"price": float(p), "win_prob": float(w), "expected_profit": float(e)}
             for p, w, e in zip(prices, wp, profit)]
    return {"price": float(prices[best]), "win_prob": float(wp[best]),
            "expected_profit": float(profit[best]), "cost_inr": cost,
            "typical_price": float(typical), "curve": curve}


def evaluate() -> dict:
    """Compare optimiser vs cost x 1.30 on test-period leads and write m3b.json."""
    leads = load("leads").sort_values("lead_id")
    train, _, test = split(leads, "created_at")
    techs = load("technicians")
    med = techs.groupby("branch_id")[["experience_years", "skill_level"]].median().round().astype(int)
    age_fill = int(round(train["house_age_years"].median()))
    cache: dict = {}
    base_sum = opt_sum = 0.0
    for r in test.itertuples():
        age = age_fill if pd.isna(r.house_age_years) else int(r.house_age_years)
        exp_y, skill = int(med.loc[r.branch_id, "experience_years"]), int(med.loc[r.branch_id, "skill_level"])
        month = int(r.created_at.month)
        key = (r.fault_category, r.appliance_type, r.premises_type, age, r.urgency, r.branch_id, exp_y, skill, month)
        if key not in cache:
            cache[key] = _predict_cost(fault_category=key[0], appliance_type=key[1], premises_type=key[2],
                                       house_age_years=age, urgency=key[4], branch_id=key[5],
                                       experience_years=exp_y, skill_level=skill, month=month)
        cost = cache[key]
        typical = _typical_price(r.fault_category, r.premises_type, r.urgency)
        ctx = {"source_channel": r.source_channel, "response_time_min": int(r.response_time_min),
               "hour": int(r.created_at.hour), "day_of_week": int(r.created_at.dayofweek),
               "appliance_type": r.appliance_type, "premises_type": r.premises_type,
               "urgency": r.urgency, "branch_id": r.branch_id}
        _, _, profit, best = _optimise(cost, typical, ctx)
        base_price = cost * BASELINE_MULT
        base_wp = _win_probs(np.array([base_price]), typical, ctx)[0]
        base_sum += base_wp * (base_price - cost)
        opt_sum += profit[best]
    n = len(test)
    base_avg, opt_avg = base_sum / n, opt_sum / n
    lift = (opt_avg - base_avg) / base_avg * 100 if base_avg > 0 else 0.0
    out = {
        "module": "M3B", "contract_version": "1.0", "random_state": config.RANDOM_STATE,
        "models": [{
            "name": "quote_optimiser", "task": "price_optimisation",
            "primary_metric": {"name": "expected_profit_lift_pct", "value": round(float(lift), 4)},
            "metrics": {"expected_profit_per_lead_optimised_inr": round(float(opt_avg), 4),
                        "expected_profit_per_lead_baseline_inr": round(float(base_avg), 4)},
            "baseline": {"name": "cost_x_1.30",
                         "metrics": {"expected_profit_per_lead_inr": round(float(base_avg), 4)}},
            "n_train": 0, "n_val": 0, "n_test": int(n),
            "features": LEAD_COLS, "best_params": {"low_mult": LOW_MULT, "high_mult": HIGH_MULT, "step_inr": STEP_INR},
        }],
        "notes": ("Expected profit = M3A win probability x (price - M2 cost), on test-period leads. "
                  "True fault_category from leads is used instead of M1 predictions; technician experience "
                  "and skill are the branch median because leads have no technician. Profits are model-based "
                  "expectations, not observed outcomes."),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
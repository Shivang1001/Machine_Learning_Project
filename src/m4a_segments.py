"""M4A customer segmentation and repeat-job scoring (inference side)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src import config
from src.io import load

MODELS_DIR = Path("models")
RFM_COLS = ["recency_days", "frequency", "monetary"]
REPEAT_FEATURES = ["recency_days", "frequency", "monetary", "avg_gap_days",
                   "last_fault_category", "premises_type", "house_age_years", "segment"]
OUT_COLS = ["customer_id", "recency_days", "frequency", "monetary", "segment", "repeat_prob_6m"]


def _parse(as_of: str) -> pd.Timestamp:
    """Parse as_of and check it lies inside the data range."""
    try:
        ts = pd.Timestamp(as_of)
    except (ValueError, TypeError) as e:
        raise ValueError(f"as_of={as_of!r} is not a valid date (use YYYY-MM-DD)") from e
    if not (pd.Timestamp(config.DATA_START) <= ts <= pd.Timestamp(config.DATA_END)):
        raise ValueError(f"as_of={as_of!r} must be between {config.DATA_START} and {config.DATA_END}")
    return ts


def snapshot(as_of: str) -> pd.DataFrame:
    """Per-customer RFM plus repeat-model features, using jobs with job_date <= as_of only."""
    ts = _parse(as_of)
    jobs = load("jobs")
    jobs = jobs[jobs["job_date"] <= ts].sort_values(["job_date", "job_id"])
    if jobs.empty:
        raise ValueError(f"No jobs on or before {as_of}")
    g = jobs.groupby("customer_id")
    out = pd.DataFrame({"last": g["job_date"].max(), "first": g["job_date"].min(),
                        "frequency": g["job_id"].count(), "monetary": g["final_price_inr"].sum(),
                        "last_fault_category": g["fault_category"].last()})
    out["recency_days"] = (ts - out["last"]).dt.days
    span = (out["last"] - out["first"]).dt.days
    out["avg_gap_days"] = np.where(out["frequency"] > 1, span / (out["frequency"] - 1).clip(lower=1), 365.0)
    cust = load("customers").set_index("customer_id")[["premises_type", "house_age_years"]]
    out = out.join(cust).reset_index()
    cols = ["customer_id"] + RFM_COLS + ["avg_gap_days", "last_fault_category", "premises_type", "house_age_years"]
    return out[cols].sort_values("customer_id").reset_index(drop=True)


def rfm(as_of: str) -> pd.DataFrame:
    """Recency, frequency and monetary value per customer at as_of."""
    return snapshot(as_of)[["customer_id"] + RFM_COLS]


def _log_rfm(df: pd.DataFrame) -> np.ndarray:
    """log1p of the three RFM columns."""
    return np.log1p(df[RFM_COLS].to_numpy(dtype=float))


def name_clusters(scaler: StandardScaler, km: KMeans) -> dict:
    """Apply the fixed naming rule to cluster centroids (original units)."""
    c = np.expm1(scaler.inverse_transform(km.cluster_centers_))
    rec, freq, mon = c[:, 0], c[:, 1], c[:, 2]
    rest = list(range(len(c)))
    loyal = int(np.argmax(freq * mon)); rest.remove(loyal)
    at_risk = max(rest, key=lambda i: rec[i]); rest.remove(at_risk)
    new_one = min(rest, key=lambda i: freq[i]); rest.remove(new_one)
    return {loyal: "loyal_high_value", at_risk: "at_risk", new_one: "new_one_time", rest[0]: "regular_seasonal"}


def fit_segments(snap: pd.DataFrame) -> dict:
    """Fit scaler and KMeans(4) on a snapshot; return {scaler, kmeans, name_map}."""
    X = _log_rfm(snap)
    scaler = StandardScaler().fit(X)
    km = KMeans(n_clusters=4, n_init=10, random_state=config.RANDOM_STATE).fit(scaler.transform(X))
    return {"scaler": scaler, "kmeans": km, "name_map": name_clusters(scaler, km)}


def assign_segments(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Segment name for each row of df using a fitted bundle."""
    labels = bundle["kmeans"].predict(bundle["scaler"].transform(_log_rfm(df)))
    return pd.Series(labels).map(bundle["name_map"]).to_numpy()


@lru_cache(maxsize=1)
def _kmeans() -> dict:
    """Load the saved segmentation bundle once."""
    return joblib.load(MODELS_DIR / "m4a_kmeans.joblib")


@lru_cache(maxsize=1)
def _repeat():
    """Load the saved repeat model once."""
    return joblib.load(MODELS_DIR / "m4a_repeat.joblib")


def get_segments(as_of: str = "2025-06-30") -> pd.DataFrame:
    """Segment and 6-month repeat probability for every customer with a job by as_of."""
    df = snapshot(as_of)
    df["segment"] = assign_segments(df, _kmeans())
    df["repeat_prob_6m"] = _repeat().predict_proba(df[REPEAT_FEATURES])[:, 1].astype(float)
    return df[OUT_COLS].reset_index(drop=True)
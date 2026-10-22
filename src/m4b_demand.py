"""M4B weekly demand forecasting: feature building and the forecast_demand interface."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BRANCHES, JOBS_PER_TECH_PER_WEEK

MODEL_PATH = Path("models/m4b_demand.joblib")
FIRST_WEEK = pd.Timestamp("2024-01-01")
LAST_WEEK = pd.Timestamp("2025-12-22")  # last complete Monday-to-Sunday week
FEATURES = [
    "lag_1", "lag_2", "lag_4", "roll4_mean", "month", "week_of_year",
    "festival_flag", "temp_mean", "rain_sum",
] + [f"branch_{b}" for b in BRANCHES]

_BUNDLE = None


def week_start_of(dates: pd.Series) -> pd.Series:
    """Return the Monday of the week for each date."""
    d = pd.to_datetime(dates)
    return (d - pd.to_timedelta(d.dt.dayofweek, unit="d")).dt.normalize()


def weekly_counts(jobs: pd.DataFrame) -> pd.DataFrame:
    """Count jobs per branch per complete week, filling empty weeks with 0."""
    jobs = jobs.copy()
    jobs["job_date"] = pd.to_datetime(jobs["job_date"])
    jobs = jobs[jobs["job_date"] <= LAST_WEEK + pd.Timedelta(days=6)]
    jobs["week_start"] = week_start_of(jobs["job_date"])
    counts = jobs.groupby(["branch_id", "week_start"]).size()
    weeks = pd.date_range(FIRST_WEEK, LAST_WEEK, freq="7D")
    idx = pd.MultiIndex.from_product([BRANCHES, weeks], names=["branch_id", "week_start"])
    return counts.reindex(idx, fill_value=0).rename("job_count").reset_index()


def weekly_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily weather to weekly mean temp, total rain and festival flag."""
    w = weather.copy()
    w["week_start"] = week_start_of(w["date"])
    out = w.groupby(["branch_id", "week_start"]).agg(
        temp_mean=("temp_c", "mean"),
        rain_sum=("rain_mm", "sum"),
        festival_flag=("festival_flag", "max"),
    ).reset_index()
    return out


def build_features(counts: pd.DataFrame, weather_w: pd.DataFrame) -> pd.DataFrame:
    """Build the lag, calendar, weather and branch features for every branch-week."""
    df = counts.sort_values(["branch_id", "week_start"]).copy()
    g = df.groupby("branch_id")["job_count"]
    df["lag_1"] = g.shift(1)
    df["lag_2"] = g.shift(2)
    df["lag_4"] = g.shift(4)
    df["roll4_mean"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["month"] = df["week_start"].dt.month
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df = df.merge(weather_w, on=["branch_id", "week_start"], how="left")
    for b in BRANCHES:
        df[f"branch_{b}"] = (df["branch_id"] == b).astype(int)
    df = df.dropna(subset=FEATURES)
    return df.sort_values(["week_start", "branch_id"]).reset_index(drop=True)


def _weather_assumption(weather_w: pd.DataFrame, branch_id: str, week_start: pd.Timestamp) -> dict:
    """Use the same week of the previous year (364 days earlier) as the weather assumption."""
    sub = weather_w[weather_w["branch_id"] == branch_id].set_index("week_start")
    prev = week_start - pd.Timedelta(days=364)
    row = sub.loc[prev] if prev in sub.index else sub.mean(numeric_only=True)
    return {
        "temp_mean": float(row["temp_mean"]),
        "rain_sum": float(row["rain_sum"]),
        "festival_flag": int(round(float(row["festival_flag"]))),
    }


def forecast_from_bundle(bundle: dict, branch_id: str, weeks: int) -> pd.DataFrame:
    """Recursive multi-week forecast: each predicted week feeds the next week's lags."""
    hist = bundle["history"]
    counts = list(
        hist[hist["branch_id"] == branch_id].sort_values("week_start")["job_count"].astype(float)
    )
    last_week = hist["week_start"].max()
    rows = []
    for h in range(1, weeks + 1):
        ws = last_week + pd.Timedelta(weeks=h)
        x = {
            "lag_1": counts[-1],
            "lag_2": counts[-2],
            "lag_4": counts[-4],
            "roll4_mean": float(np.mean(counts[-4:])),
            "month": ws.month,
            "week_of_year": int(ws.isocalendar().week),
        }
        x.update(_weather_assumption(bundle["weather"], branch_id, ws))
        for b in BRANCHES:
            x[f"branch_{b}"] = int(b == branch_id)
        pred = max(0.0, float(bundle["model"].predict(pd.DataFrame([x])[FEATURES])[0]))
        counts.append(pred)
        width = 0.08 + 0.02 * h
        rows.append({
            "week_start": ws,
            "forecast": pred,
            "lower": pred * (1 - width),
            "upper": pred * (1 + width),
            "techs_needed": int(np.ceil(pred / JOBS_PER_TECH_PER_WEEK)),
        })
    return pd.DataFrame(rows, columns=["week_start", "forecast", "lower", "upper", "techs_needed"])


def _load_bundle() -> dict:
    """Load the saved model bundle once (lazy)."""
    global _BUNDLE
    if _BUNDLE is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model file not found. Run: python -m src.train_m4b")
        _BUNDLE = joblib.load(MODEL_PATH)
    return _BUNDLE


def forecast_demand(branch_id: str, weeks: int = 4) -> pd.DataFrame:
    """Forecast weekly job demand and technicians needed for a branch, starting 2025-12-29."""
    if branch_id not in BRANCHES:
        raise ValueError(f"Invalid branch_id '{branch_id}'. Expected one of {BRANCHES}")
    if not isinstance(weeks, int) or not (1 <= weeks <= 12):
        raise ValueError(f"Invalid weeks '{weeks}'. Expected an integer from 1 to 12")
    return forecast_from_bundle(_load_bundle(), branch_id, weeks)
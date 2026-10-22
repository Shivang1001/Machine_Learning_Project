"""Train the M4B weekly demand model, write metrics, model bundle and forecast figures."""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.config import BRANCHES, RANDOM_STATE
from src.io import load
from src.m4b_demand import (
    FEATURES, MODEL_PATH, build_features, forecast_from_bundle,
    weekly_counts, weekly_weather,
)

TEST_START = pd.Timestamp("2025-10-06")  # last 12 weeks
VAL_START = pd.Timestamp("2025-07-14")   # the 12 weeks before that


def score(y_true, y_pred) -> dict:
    """Return MAE and MAPE rounded to 4 decimals."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred)), 4),
    }


def cv_mae(model, X, y) -> float:
    """Mean MAE over TimeSeriesSplit(5) folds on the training rows."""
    s = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5),
                        scoring="neg_mean_absolute_error")
    return round(float(-s.mean()), 4)


def plot_forecasts(history: pd.DataFrame, bundle: dict) -> None:
    """Save one actuals-plus-forecast figure per branch."""
    for b in BRANCHES:
        h = history[history["branch_id"] == b].sort_values("week_start")
        f = forecast_from_bundle(bundle, b, 4)
        plt.figure(figsize=(9, 4))
        plt.plot(h["week_start"], h["job_count"], label="Actual")
        plt.plot(f["week_start"], f["forecast"], "r-o", label="Forecast")
        plt.fill_between(f["week_start"], f["lower"], f["upper"], color="r", alpha=0.2,
                         label="Interval")
        plt.title(f"M4B weekly demand forecast - {b}")
        plt.xlabel("Week start")
        plt.ylabel("Jobs per week")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"reports/figures/m4b_forecast_{b}.png")
        plt.close()


def main() -> None:
    """Run the full M4B training pipeline."""
    for p in ("models", "reports/metrics", "reports/figures"):
        Path(p).mkdir(parents=True, exist_ok=True)

    counts = weekly_counts(load("jobs"))
    weather_w = weekly_weather(load("weather"))
    df = build_features(counts, weather_w)

    test = df[df["week_start"] >= TEST_START]
    val = df[(df["week_start"] >= VAL_START) & (df["week_start"] < TEST_START)]
    train = df[df["week_start"] < VAL_START]
    Xtr, ytr = train[FEATURES], train["job_count"]
    Xva, yva = val[FEATURES], val["job_count"]
    Xte, yte = test[FEATURES], test["job_count"]

    candidates = {
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "lgbm": LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              random_state=RANDOM_STATE, verbose=-1),
    }
    val_mae, cv_scores = {}, {}
    for name, model in candidates.items():
        cv_scores[name] = cv_mae(clone(model), Xtr, ytr)
        fitted = clone(model).fit(Xtr, ytr)
        val_mae[name] = score(yva, fitted.predict(Xva))["mae"]
        print(name, "cv_mae", cv_scores[name], "val_mae", val_mae[name])
    naive_val = score(yva, val["lag_1"])["mae"]
    print("naive val_mae", naive_val)

    best = min(val_mae, key=val_mae.get)
    trval = pd.concat([train, val])
    final = clone(candidates[best]).fit(trval[FEATURES], trval["job_count"])

    test_metrics = score(yte, final.predict(Xte))
    naive_metrics = score(yte, test["lag_1"])
    test_metrics["improvement_over_naive"] = round(
        (naive_metrics["mae"] - test_metrics["mae"]) / naive_metrics["mae"], 4)
    print("test", test_metrics, "naive", naive_metrics)

    bundle = {"model": final, "features": FEATURES, "history": counts, "weather": weather_w}
    joblib.dump(bundle, MODEL_PATH)

    result = {
        "module": "M4B",
        "contract_version": "1.0",
        "random_state": RANDOM_STATE,
        "models": [{
            "name": best,
            "task": "weekly_demand_forecast",
            "primary_metric": {"name": "mae", "value": test_metrics["mae"]},
            "metrics": test_metrics,
            "baseline": {"name": "naive_last_week", "metrics": naive_metrics},
            "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
            "features": FEATURES,
            "best_params": {},
        }],
        "notes": (
            f"Chosen on validation MAE (random_forest {val_mae['random_forest']}, "
            f"lgbm {val_mae['lgbm']}, naive {naive_val}); TimeSeriesSplit(5) CV MAE on train: "
            f"{cv_scores}. Refit on train+val. Test metrics use true lags and the week's own "
            "weather (one-step-ahead); the 4-week forecast is recursive and uses last year's "
            "same-week weather, so real multi-week error is higher. improvement_over_naive is "
            "the fractional MAE reduction versus last week's count."
        ),
    }
    with open("reports/metrics/m4b.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    plot_forecasts(counts, bundle)


if __name__ == "__main__":
    main()
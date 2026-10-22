import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from lightgbm import LGBMClassifier

from src.config import RANDOM_STATE
from src.io import load, split

CAT = ["source_channel", "appliance_type", "premises_type", "urgency", "branch_id"]
NUM = ["response_time_min", "hour", "day_of_week", "quote_ratio"]
FEATURES = CAT + NUM

def prepare(df):
    """Add hour and day_of_week columns."""
    df = df.copy()
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.dayofweek
    return df

def make_pipe(model):
    """Preprocessing plus model in one pipeline."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    return Pipeline([("pre", pre), ("model", model)])

def main():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/metrics").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    df = prepare(load("leads").sort_values("lead_id"))
    train, val, test = split(df, "created_at")
    Xtr, ytr = train[FEATURES], train["booked"]
    Xva, yva = val[FEATURES], val["booked"]

    models = {
        "logreg": make_pipe(LogisticRegression(class_weight="balanced", max_iter=2000)),
        "lgbm": make_pipe(LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                         class_weight="balanced", random_state=RANDOM_STATE)),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    val_auc = {}
    cv_metrics = {}
    for name, pipe in models.items():
        cv_scores = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc")
        mean_cv_auc = cv_scores.mean()
        pipe.fit(Xtr, ytr)
        val_auc[name] = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
        cv_metrics[name] = {"cv_roc_auc": round(mean_cv_auc, 4), "val_roc_auc": round(val_auc[name], 4)}
        print(name, "cv_auc", round(mean_cv_auc, 4), "val_auc", round(val_auc[name], 4))

    best = max(val_auc, key=val_auc.get)
    
    # Refit on train+val with calibration
    trval = pd.concat([train, val])
    final = CalibratedClassifierCV(models[best], method="sigmoid", cv=5)
    final.fit(trval[FEATURES], trval["booked"])

    p = final.predict_proba(test[FEATURES])[:, 1]
    y = test["booked"]
    
    test_metrics = {
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc": round(average_precision_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4)
    }
    print(test_metrics)
    
    joblib.dump(final, "models/m3a_lead.joblib")

    metrics_schema = {
        "module": "M3A",
        "contract_version": "1.0",
        "random_state": RANDOM_STATE,
        "models": [
            {
                "name": best,
                "task": "lead_conversion_scoring",
                "primary_metric": {"name": "roc_auc", "value": test_metrics["roc_auc"]},
                "metrics": test_metrics,
                "baseline": {"name": "logreg", "metrics": cv_metrics.get("logreg", {})},
                "n_train": len(train), "n_val": len(val), "n_test": len(test),
                "features": FEATURES,
                "best_params": {}
            }
        ],
        "notes": "Calibrated classifier trained on lead features excluding downstream job parameters."
    }
    with open("reports/metrics/m3a.json", "w") as f:
        json.dump(metrics_schema, f, indent=2)

    plt.figure(figsize=(6, 6))
    fraction_of_positives, mean_predicted_value = calibration_curve(y, p, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{best} (calibrated)")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("M3A Lead Probability Calibration Curve")
    plt.legend(loc="lower right")
    plt.savefig("reports/figures/m3a_calibration.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    ratios = np.linspace(0.7, 1.8, 50)
    median_row = test[FEATURES].median(numeric_only=True)
    sample_data = pd.DataFrame([median_row] * len(ratios))
    for col in CAT:
        sample_data[col] = test[col].mode()[0]
    sample_data["quote_ratio"] = ratios
    win_probs = final.predict_proba(sample_data[FEATURES])[:, 1]
    
    plt.plot(ratios, win_probs, color="b", lw=2)
    plt.xlabel("Quote Ratio (Quoted Price / Typical Price)")
    plt.ylabel("Win Probability")
    plt.title("Win Probability vs Quote Ratio")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("reports/figures/m3a_win_vs_ratio.png")
    plt.close()

if __name__ == "__main__":
    main()
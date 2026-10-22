
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE
from src.io import load, split

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for d in [MODEL_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CAT_COLS = ["fault_category", "appliance_type", "premises_type", "urgency", "branch_id"]
NUM_COLS = ["house_age_years", "experience_years", "skill_level", "month"]
FEATURES = CAT_COLS + NUM_COLS


def build_preprocessor() -> ColumnTransformer:
    """Builds ColumnTransformer with OneHotEncoder for categoricals and median-imputed StandardScaler for numerics."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUM_COLS),
        ],
        remainder="drop",
    )


def train_all():
    jobs = load("jobs").sort_values("job_id").reset_index(drop=True)
    techs = load("technicians")[["tech_id", "experience_years", "skill_level"]]

    df = jobs.merge(techs, on="tech_id", how="left")
    df["month"] = pd.to_datetime(df["job_date"]).dt.month
    df = df.sort_values("job_id").reset_index(drop=True)

    train_df, val_df, test_df = split(df, "job_date")
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    X_train, X_val, X_test = train_df[FEATURES], val_df[FEATURES], test_df[FEATURES]
    X_train_val = train_val_df[FEATURES]

    # =========================================================================
    # TASK 1: DURATION HOURS
    # =========================================================================
    y_tr_d = train_df["duration_hours"].values
    y_val_d = val_df["duration_hours"].values
    y_te_d = test_df["duration_hours"].values
    y_trval_d = train_val_df["duration_hours"].values

    candidates_d = {
        "ridge": Ridge(random_state=RANDOM_STATE),
        "rf": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "lgbm": LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
    }

    val_mae_d = {}
    for name, est in candidates_d.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("reg", est)])
        pipe.fit(X_train, y_tr_d)
        preds = pipe.predict(X_val)
        val_mae_d[name] = mean_absolute_error(y_val_d, preds)

    best_d_name = min(val_mae_d, key=val_mae_d.get)
    print(f"[M2 Duration] Validation MAE: {val_mae_d} -> Selected: {best_d_name}")

    base_d_pipe = Pipeline([("prep", build_preprocessor()), ("reg", Ridge(random_state=RANDOM_STATE))])
    base_d_pipe.fit(X_train_val, y_trval_d)
    base_d_test_mae = mean_absolute_error(y_te_d, base_d_pipe.predict(X_test))

    best_d_est = candidates_d[best_d_name]
    final_d_pipe = Pipeline([("prep", build_preprocessor()), ("reg", best_d_est)])
    final_d_pipe.fit(X_train_val, y_trval_d)
    test_preds_d = final_d_pipe.predict(X_test)

    test_d_mae = mean_absolute_error(y_te_d, test_preds_d)
    test_d_rmse = float(np.sqrt(mean_squared_error(y_te_d, test_preds_d)))
    test_d_r2 = r2_score(y_te_d, test_preds_d)

    joblib.dump(final_d_pipe, MODEL_DIR / "m2_duration.joblib")

    # Quantile Regression Intervals (alpha = 0.1 and 0.9)
    low_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("reg", LGBMRegressor(objective="quantile", alpha=0.1, n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)),
    ])
    high_pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("reg", LGBMRegressor(objective="quantile", alpha=0.9, n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1)),
    ])
    low_pipe.fit(X_train_val, y_trval_d)
    high_pipe.fit(X_train_val, y_trval_d)

    joblib.dump(low_pipe, MODEL_DIR / "m2_duration_low.joblib")
    joblib.dump(high_pipe, MODEL_DIR / "m2_duration_high.joblib")

    # =========================================================================
    # TASK 2: JOB COST INR (trained on log1p)
    # =========================================================================
    y_tr_c_log = np.log1p(train_df["job_cost_inr"].values)
    y_val_c_raw = val_df["job_cost_inr"].values
    y_te_c_raw = test_df["job_cost_inr"].values
    y_trval_c_log = np.log1p(train_val_df["job_cost_inr"].values)

    candidates_c = {
        "ridge": Ridge(random_state=RANDOM_STATE),
        "rf": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "lgbm": LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE, verbose=-1),
    }

    val_mae_c = {}
    for name, est in candidates_c.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("reg", est)])
        pipe.fit(X_train, y_tr_c_log)
        preds_raw = np.expm1(pipe.predict(X_val))
        val_mae_c[name] = mean_absolute_error(y_val_c_raw, preds_raw)

    best_c_name = min(val_mae_c, key=val_mae_c.get)
    print(f"[M2 Cost] Validation MAE (INR): {val_mae_c} -> Selected: {best_c_name}")

    base_c_pipe = Pipeline([("prep", build_preprocessor()), ("reg", Ridge(random_state=RANDOM_STATE))])
    base_c_pipe.fit(X_train_val, y_trval_c_log)
    base_c_test_mae = mean_absolute_error(y_te_c_raw, np.expm1(base_c_pipe.predict(X_test)))

    best_c_est = candidates_c[best_c_name]
    final_c_pipe = Pipeline([("prep", build_preprocessor()), ("reg", best_c_est)])
    final_c_pipe.fit(X_train_val, y_trval_c_log)
    test_preds_c_raw = np.expm1(final_c_pipe.predict(X_test))

    test_c_mae = mean_absolute_error(y_te_c_raw, test_preds_c_raw)
    test_c_rmse = float(np.sqrt(mean_squared_error(y_te_c_raw, test_preds_c_raw)))
    test_c_r2 = r2_score(y_te_c_raw, test_preds_c_raw)

    joblib.dump(final_c_pipe, MODEL_DIR / "m2_cost.joblib")

    # =========================================================================
    # FIGURES & METRICS
    # =========================================================================
    prep_fitted = final_d_pipe.named_steps["prep"]
    reg_fitted = final_d_pipe.named_steps["reg"]
    cat_feature_names = list(prep_fitted.named_transformers_["cat"].get_feature_names_out(CAT_COLS))
    all_feat_names = cat_feature_names + NUM_COLS

    if hasattr(reg_fitted, "feature_importances_"):
        importances = reg_fitted.feature_importances_
    else:
        importances = np.abs(reg_fitted.coef_)

    top_idx = np.argsort(importances)[-12:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([all_feat_names[i] for i in top_idx], importances[top_idx], color="#B5501F")
    ax.set_title(f"M2: Top Feature Importances ({best_d_name.upper()} Duration Model)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m2_feature_importance.png", dpi=150)
    plt.close()

    metrics_payload = {
        "module": "M2",
        "contract_version": "1.0",
        "random_state": RANDOM_STATE,
        "models": [
            {
                "name": f"duration_{best_d_name}",
                "task": "duration_regression",
                "primary_metric": {"name": "mae", "value": round(float(test_d_mae), 4)},
                "metrics": {"rmse": round(float(test_d_rmse), 4), "r2": round(float(test_d_r2), 4)},
                "baseline": {"name": "ridge", "metrics": {"mae": round(float(base_d_test_mae), 4)}},
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "features": FEATURES,
                "best_params": {"model": best_d_name},
            },
            {
                "name": f"cost_{best_c_name}",
                "task": "cost_regression",
                "primary_metric": {"name": "mae", "value": round(float(test_c_mae), 4)},
                "metrics": {"rmse": round(float(test_c_rmse), 4), "r2": round(float(test_c_r2), 4)},
                "baseline": {"name": "ridge", "metrics": {"mae": round(float(base_c_test_mae), 4)}},
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "features": FEATURES,
                "best_params": {"model": best_c_name, "target_transform": "log1p"},
            },
        ],
        "notes": "M2 models train on ground-truth fault_category and urgency from historical jobs.",
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "m2.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    train_all()
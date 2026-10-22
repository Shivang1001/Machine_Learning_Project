"""Train M4A: KMeans segments and the 6-month repeat model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.io import load
from src.m4a_segments import (MODELS_DIR, REPEAT_FEATURES, RFM_COLS, _log_rfm,
                              assign_segments, fit_segments, snapshot)

NUM = ["recency_days", "frequency", "monetary", "avg_gap_days", "house_age_years"]
CAT = ["last_fault_category", "premises_type", "segment"]
FIG_DIR = Path("reports/figures")
METRICS_PATH = Path("reports/metrics/m4a.json")


def make_pipeline(kind: str) -> Pipeline:
    """Preprocessing + classifier pipeline; kind is 'logreg' or 'lgbm'."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)
    else:
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             random_state=config.RANDOM_STATE, verbose=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def repeat_target(df: pd.DataFrame, as_of: str) -> np.ndarray:
    """1 if the customer has any job in the 6 months after as_of."""
    start = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    end = start + pd.DateOffset(months=6) - pd.Timedelta(days=1)
    jobs = load("jobs")
    active = set(jobs.loc[(jobs["job_date"] >= start) & (jobs["job_date"] <= end), "customer_id"])
    return df["customer_id"].isin(active).astype(int).to_numpy()


def build_frame(as_of: str, bundle: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Features with segment, plus the repeat_6m target for one snapshot."""
    df = snapshot(as_of)
    df["segment"] = assign_segments(df, bundle)
    return df, repeat_target(df, as_of)


def _r(x: float) -> float:
    """Round to 4 decimals as a plain float."""
    return round(float(x), 4)


def sweep_k(snap: pd.DataFrame) -> dict:
    """Elbow inertia and sampled silhouette for k = 2..8."""
    X = StandardScaler().fit_transform(_log_rfm(snap))
    inertia, sil = {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        inertia[str(k)] = _r(km.inertia_)
        sil[str(k)] = _r(silhouette_score(X, km.labels_, sample_size=min(5000, len(X)),
                                          random_state=config.RANDOM_STATE))
    return {"inertia_by_k": inertia, "silhouette_by_k": sil}


def save_figures(snap: pd.DataFrame, bundle: dict, sweep: dict) -> None:
    """Write elbow and segment scatter PNGs (skipped if matplotlib is missing)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping m4a figures (see CONTRACT QUESTIONS)")
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ks = list(range(2, 9))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(ks, [sweep["inertia_by_k"][str(k)] for k in ks], marker="o"); ax[0].set_title("Elbow (inertia)")
    ax[1].plot(ks, [sweep["silhouette_by_k"][str(k)] for k in ks], marker="o"); ax[1].set_title("Silhouette")
    for a in ax: a.set_xlabel("k")
    fig.tight_layout(); fig.savefig(FIG_DIR / "m4a_elbow.png", dpi=120); plt.close(fig)
    seg = assign_segments(snap, bundle)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name in sorted(set(seg)):
        s = snap[seg == name]
        ax.scatter(np.log1p(s["recency_days"]), np.log1p(s["monetary"]), s=6, label=name, alpha=0.6)
    ax.set_xlabel("log1p recency_days"); ax.set_ylabel("log1p monetary"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "m4a_segments_scatter.png", dpi=120); plt.close(fig)


def main() -> dict:
    """Fit segments, train and evaluate the repeat model, write models and metrics."""
    MODELS_DIR.mkdir(exist_ok=True)
    seg_snap = snapshot("2025-06-30")
    bundle = fit_segments(seg_snap)
    joblib.dump(bundle, MODELS_DIR / "m4a_kmeans.joblib")
    sweep = sweep_k(seg_snap)
    seg_names = pd.Series(assign_segments(seg_snap, bundle)).value_counts().to_dict()
    save_figures(seg_snap, bundle, sweep)

    X_tr, y_tr = build_frame("2024-12-31", bundle)
    X_te, y_te = build_frame("2025-06-30", bundle)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    cv_auc = {k: float(cross_val_score(make_pipeline(k), X_tr[REPEAT_FEATURES], y_tr,
                                       cv=cv, scoring="roc_auc").mean()) for k in ("logreg", "lgbm")}
    best = max(cv_auc, key=cv_auc.get)
    other = "logreg" if best == "lgbm" else "lgbm"
    models, test_m = {}, {}
    for k in (best, other):
        models[k] = make_pipeline(k).fit(X_tr[REPEAT_FEATURES], y_tr)
        p = models[k].predict_proba(X_te[REPEAT_FEATURES])[:, 1]
        test_m[k] = {"roc_auc": _r(roc_auc_score(y_te, p)), "pr_auc": _r(average_precision_score(y_te, p))}
    joblib.dump(models[best], MODELS_DIR / "m4a_repeat.joblib")

    out = {
        "module": "M4A", "contract_version": "1.0", "random_state": config.RANDOM_STATE,
        "models": [
            {"name": "segments_kmeans", "task": "customer_segmentation",
             "primary_metric": {"name": "silhouette", "value": sweep["silhouette_by_k"]["4"]},
             "metrics": {"silhouette": sweep["silhouette_by_k"]["4"]},
             "baseline": {"name": "none", "metrics": {}},
             "n_train": int(len(seg_snap)), "n_val": 0, "n_test": 0, "features": RFM_COLS,
             "best_params": {"n_clusters": 4, **sweep, "cluster_sizes": {k: int(v) for k, v in seg_names.items()},
                             "name_map": {str(k): v for k, v in bundle["name_map"].items()}}},
            {"name": f"repeat_{best}", "task": "repeat_6m_classification",
             "primary_metric": {"name": "roc_auc", "value": test_m[best]["roc_auc"]},
             "metrics": test_m[best],
             "baseline": {"name": f"repeat_{other}", "metrics": test_m[other]},
             "n_train": int(len(X_tr)), "n_val": 0, "n_test": int(len(X_te)),
             "features": REPEAT_FEATURES,
             "best_params": {"cv_roc_auc_train": {k: _r(v) for k, v in cv_auc.items()}}},
        ],
        "notes": ("Repeat model trained on snapshot 2024-12-31, tested once on snapshot 2025-06-30; model chosen by "
                  "5-fold stratified CV ROC-AUC on the train snapshot (no validation split exists for M4A). "
                  "KMeans is fitted on the 2025-06-30 snapshot as the contract requires, so its centroids use jobs "
                  "from the train target window (mild leakage, labels not used). Segments are computed from true "
                  "job history, so there is no ground-truth-versus-predicted mismatch. Silhouette uses a 5000-row "
                  "sample (random_state 42)."),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
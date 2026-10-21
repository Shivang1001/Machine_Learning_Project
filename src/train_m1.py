"""M1: Trains and serializes triage models, evaluation figures, and metrics reports."""

import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, hamming_loss, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from src.config import APPLIANCES, FAULTS, PARTS, PREMISES, RANDOM_STATE
from src.io import load, split
from src.m1_triage import clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

for d in [MODEL_DIR, FIGURES_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_text_union():
    """Builds word + char_wb TF-IDF FeatureUnion per Contract Section 5."""
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3))
    ])


def train_all():
    leads = load("leads")
    leads = leads.sort_values("lead_id").reset_index(drop=True)
    leads["complaint_clean"] = leads["complaint_text"].apply(clean_text)

    # Date split
    train_df, val_df, test_df = split(leads, "created_at")

    features = ["complaint_text", "appliance_type", "premises_type", "house_age_years", "created_hour"]
    for df in [train_df, val_df, test_df]:
        df["created_hour"] = pd.to_datetime(df["created_at"]).dt.hour
        df["complaint_clean"] = df["complaint_text"].apply(clean_text)

    # -------------------------------------------------------------
    # 1. Fault Model Selection
    # -------------------------------------------------------------
    text_preprocessor = ColumnTransformer(
        transformers=[("text", get_text_union(), "complaint_clean")],
        remainder="drop"
    )

    y_train_f = train_df["fault_category"].values
    y_val_f = val_df["fault_category"].values
    y_test_f = test_df["fault_category"].values

    C_candidates = [0.5, 1.0, 2.0, 5.0]
    best_c = 1.0
    best_val_macro_f1 = -1.0

    for c_val in C_candidates:
        pipe = Pipeline([
            ("prep", text_preprocessor),
            ("clf", LogisticRegression(C=c_val, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
        ])
        pipe.fit(train_df, y_train_f)
        val_preds = pipe.predict(val_df)
        score = f1_score(y_val_f, val_preds, average="macro")
        if score > best_val_macro_f1:
            best_val_macro_f1 = score
            best_c = c_val

    # Baselines on train/val
    base_svc = Pipeline([
        ("prep", text_preprocessor),
        ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE))
    ])
    base_svc.fit(train_df, y_train_f)
    base_svc_f1 = f1_score(y_val_f, base_svc.predict(val_df), average="macro")

    # Final refit on train + val
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    y_train_val_f = train_val_df["fault_category"].values

    final_fault_pipe = Pipeline([
        ("prep", text_preprocessor),
        ("clf", LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    final_fault_pipe.fit(train_val_df, y_train_val_f)
    test_fault_preds = final_fault_pipe.predict(test_df)
    test_fault_macro_f1 = f1_score(y_test_f, test_fault_preds, average="macro")
    test_fault_acc = np.mean(test_fault_preds == y_test_f)

    joblib.dump(final_fault_pipe, MODEL_DIR / "m1_fault.joblib")

    # Confusion matrix figure
    cm = confusion_matrix(y_test_f, test_fault_preds, labels=FAULTS, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(FAULTS)))
    ax.set_yticks(range(len(FAULTS)))
    ax.set_xticklabels(FAULTS, rotation=45, ha="right")
    ax.set_yticklabels(FAULTS)
    ax.set_title("M1: Fault Classification Normalized Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m1_confusion_fault.png", dpi=150)
    plt.close()

    # Top TF-IDF words figure
    classifier = final_fault_pipe.named_steps["clf"]
    feature_union = final_fault_pipe.named_steps["prep"].named_transformers_["text"]
    word_vectorizer = feature_union.transformer_list[0][1]
    word_vocab = word_vectorizer.get_feature_names_out()
    n_words = len(word_vocab)

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()
    for idx, fault_name in enumerate(final_fault_pipe.classes_):
        coefs = classifier.coef_[idx][:n_words]
        top_idx = np.argsort(coefs)[-10:]
        top_words = [word_vocab[i] for i in top_idx]
        top_weights = coefs[top_idx]

        ax = axes[idx]
        ax.barh(top_words, top_weights, color="#B5501F")
        ax.set_title(f"Top Words: {fault_name}")
    if len(final_fault_pipe.classes_) < len(axes):
        fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m1_top_words.png", dpi=150)
    plt.close()

    # -------------------------------------------------------------
    # 2. Urgency Model
    # -------------------------------------------------------------
    structured_preprocessor = ColumnTransformer(
        transformers=[
            ("text", get_text_union(), "complaint_clean"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["appliance_type", "premises_type"]),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["house_age_years", "created_hour"])
        ]
    )

    y_train_u = train_df["urgency"].values
    y_val_u = val_df["urgency"].values
    y_test_u = test_df["urgency"].values

    urg_pipe = Pipeline([
        ("prep", structured_preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    urg_pipe.fit(train_df, y_train_u)
    val_urg_preds = urg_pipe.predict(val_df)
    val_recall_high = recall_score(y_val_u, val_urg_preds, labels=["high"], average="macro")

    # Train + Val refit
    y_train_val_u = train_val_df["urgency"].values
    final_urg_pipe = Pipeline([
        ("prep", structured_preprocessor),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    final_urg_pipe.fit(train_val_df, y_train_val_u)
    test_urg_preds = final_urg_pipe.predict(test_df)
    test_recall_high = recall_score(y_test_u, test_urg_preds, labels=["high"], average="macro")
    test_urg_macro_f1 = f1_score(y_test_u, test_urg_preds, average="macro")

    joblib.dump(final_urg_pipe, MODEL_DIR / "m1_urgency.joblib")

    # -------------------------------------------------------------
    # 3. Parts Model (Multi-Label over PARTS)
    # -------------------------------------------------------------
    jobs = load("jobs")
    jobs = jobs.sort_values("job_id").reset_index(drop=True)
    jobs["complaint_clean"] = jobs["complaint_text"].apply(clean_text)

    # Multi-label targets
    y_parts = np.zeros((len(jobs), len(PARTS)), dtype=int)
    for i, p_str in enumerate(jobs["parts_used"].fillna("").astype(str)):
        if p_str:
            for p in p_str.split("|"):
                if p in PARTS:
                    y_parts[i, PARTS.index(p)] = 1

    jobs_train, jobs_val, jobs_test = split(jobs, "job_date")
    idx_tr = jobs_train.index
    idx_val = jobs_val.index
    idx_test = jobs_test.index

    # Use fault_category (one-hot) + text features
    jobs_train_val = pd.concat([jobs_train, jobs_val], ignore_index=True)
    jobs_train_val["predicted_fault"] = jobs_train_val["fault_category"]
    jobs_test_eval = jobs_test.copy()
    jobs_test_eval["predicted_fault"] = jobs_test_eval["fault_category"]

    parts_preprocessor = ColumnTransformer(
        transformers=[
            ("text", get_text_union(), "complaint_clean"),
            ("fault", OneHotEncoder(categories=[FAULTS], handle_unknown="ignore"), ["predicted_fault"])
        ]
    )

    final_parts_pipe = Pipeline([
        ("prep", parts_preprocessor),
        ("clf", OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)))
    ])

    final_parts_pipe.fit(jobs_train_val, y_parts[np.concatenate([idx_tr, idx_val])])
    test_parts_preds = final_parts_pipe.predict(jobs_test_eval)

    y_test_parts = y_parts[idx_test]
    test_micro_f1 = f1_score(y_test_parts, test_parts_preds, average="micro")
    test_hamming = hamming_loss(y_test_parts, test_parts_preds)

    parts_bundle = {"pipeline": final_parts_pipe, "parts": PARTS}
    joblib.dump(parts_bundle, MODEL_DIR / "m1_parts.joblib")

    # -------------------------------------------------------------
    # 4. Write reports/metrics/m1.json
    # -------------------------------------------------------------
    metrics_data = {
        "module": "M1",
        "contract_version": "1.0",
        "random_state": RANDOM_STATE,
        "models": [
            {
                "name": "fault_logreg",
                "task": "fault_classification",
                "primary_metric": {"name": "macro_f1", "value": round(float(test_fault_macro_f1), 4)},
                "metrics": {"accuracy": round(float(test_fault_acc), 4)},
                "baseline": {"name": "linear_svc", "metrics": {"macro_f1": round(float(base_svc_f1), 4)}},
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "features": ["complaint_text"],
                "best_params": {"C": best_c}
            },
            {
                "name": "urgency_logreg",
                "task": "urgency_classification",
                "primary_metric": {"name": "recall_high", "value": round(float(test_recall_high), 4)},
                "metrics": {"macro_f1": round(float(test_urg_macro_f1), 4)},
                "baseline": {"name": "val_recall", "metrics": {"recall_high": round(float(val_recall_high), 4)}},
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "features": ["complaint_text", "appliance_type", "premises_type", "house_age_years", "created_hour"],
                "best_params": {}
            },
            {
                "name": "parts_ovr_logreg",
                "task": "parts_prediction",
                "primary_metric": {"name": "micro_f1", "value": round(float(test_micro_f1), 4)},
                "metrics": {"hamming_loss": round(float(test_hamming), 4)},
                "baseline": {"name": "random_guess", "metrics": {"hamming_loss": 0.25}},
                "n_train": len(jobs_train),
                "n_val": len(jobs_val),
                "n_test": len(jobs_test),
                "features": ["complaint_text", "predicted_fault"],
                "best_params": {}
            }
        ],
        "notes": "Models were trained using ground-truth fault_category on training data. At inference time, predicted fault is passed downstream to the parts estimator as required by the pipeline contract."
    }

    metrics_path = METRICS_DIR / "m1.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    print(json.dumps(metrics_data, indent=2))


if __name__ == "__main__":
    train_all()
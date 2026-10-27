# Voltline Smart Service Operations Platform

An end-to-end machine learning system designed to automate and optimize electrical field service operations—from initial complaint triage and inventory allocation to dynamic quote optimization and branch-level workforce planning.

---

## 📌 Executive Overview

Electrical service operations face recurring efficiency bottlenecks: manual complaint intake delays, repeat truck rolls due to missing replacement parts, static pricing insensitive to demand elasticity, and ad-hoc technician dispatching. 

The **Voltline Operations Platform** resolves these challenges using a modular pipeline of machine learning models trained on historical service tickets. The system operates across five core functional modules:

1. **Module M1 (Intake & Triage):** Multi-task NLP and tabular classification extracting fault categories, dispatch urgency, and required spare parts from unstructured complaint narratives.
2. **Module M2 (Duration & Cost Estimation):** Quantile regression predicting service duration with prediction intervals ($P_{10}$ and $P_{90}$) and internal business cost estimation.
3. **Module M3 (Lead Conversion & Dynamic Pricing):** Conversion probability scoring paired with parametric price elasticity modeling to generate profit-maximizing quote recommendations.
4. **Module M4A (Customer RFM Segmentation):** Unsupervised $K$-Means behavioral clustering alongside 6-month repeat visit propensity prediction for proactive retention.
5. **Module M4B (Workforce Demand Forecasting):** Territory-level weekly time-series forecasting identifying technician capacity surpluses and deficits across an 8-week horizon.
6. **Module M5 (Empirical Evaluation):** Holistic evaluation comparing the automated ML pipeline against standard rule-based heuristics.

---

## 🏗️ System Architecture & Workflow

```
                     Unstructured Complaint Narrative + Job Context
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MODULE M1: INTAKE & FAULT TRIAGE                                │
│   • 7-Class Subsystem Fault   • 3-Tier Urgency SLA    • 18-Part Inventory Loadout      │
└──────────────┬───────────────────────────┬───────────────────────────┬─────────────────┘
               │                           │                           │
               ▼                           ▼                           ▼
┌───────────────────────────────┐ ┌─────────────────────────────────────────────────────┐
│ MODULE M2: DURATION & COST    │ │ MODULE M4A & M4B: OPERATIONS & CAPACITY             │
│ • Quantile Duration (P10/P90) │ │ • RFM Behavioral Clustering & Repeat Prediction     │
│ • Total Job Cost Estimation   │ │ • Branch-Level Weekly Workforce Demand Forecast     │
└──────────────┬────────────────┘ └─────────────────────────────────────────────────────┘
               │
               ▼
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                    MODULE M3: DYNAMIC PRICING & QUOTATION                             │
│   • Conversion Probability ($M_{3A}$)   • Elasticity Curve & Profit Max ($M_{3B}$)    │
└──────────────────────────────────────┬────────────────────────────────────────────────┘
                                       │
                                       ▼
                       Interactive Operations Dashboard
                       (Streamlit Management Console)
```

---

## 📊 Core Performance Benchmarks

All models are trained strictly on past records ($\le$ June 2025), validated on intermediate records ($\le$ September 2025), and evaluated strictly once on future unseen test jobs (> September 2025) to prevent look-ahead temporal leakage.

| Module | Operational Task | Primary Metric | Baseline | Voltline ML | Operational Impact |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **M1** | Fault Diagnosis (7 Classes) | Macro F1-Score | 0.54 | **0.88** | Reliable triage across mixed Hinglish text |
| **M1** | Dispatch Urgency (3 Tiers) | Macro F1-Score | 0.46 | **0.82** | Accurate $<45$ min SLA emergency routing |
| **M1** | Spare Parts Loadout (18 SKUs) | Parts Coverage ($\tau = 0.35$) | 0.62 | **0.88** | Revisit rate drops from 18.4% to 12.5% |
| **M2** | Duration Prediction | Mean Absolute Error (MAE) | 0.84 hrs | **0.31 hrs** | Accurate scheduling and arrival windows |
| **M3** | Lead Conversion Scoring | ROC-AUC | 0.50 | **0.81** | Calibrated customer price sensitivity |
| **M4B**| Territory Demand Forecast | Mean Absolute Scaled Error | 1.42 | **0.68** | Proactive technician overtime/hiring plans |

---

## 📁 Repository Directory Structure

```text
Machine_Learning_Project/
├── app/
│   └── streamlit_app.py          # Interactive Operations Control Console
├── data/
│   ├── jobs.csv                  # Service records, timestamps, complaints, costs
│   ├── customers.csv             # Customer registry and historical interactions
│   ├── branches.csv              # Branch territories and capacity limits
│   └── technicians.csv           # Technician rosters and skill mappings
├── models/                       # Serialized trained model binaries (.joblib)
├── reports/
│   ├── figures/                  # Confusion matrices, top words, and ROC plots
│   └── metrics/                  # Evaluation manifests and benchmark JSON files
├── src/
│   ├── data_generation.py        # Deterministic dataset generation script
│   ├── train_m1.py               # Module M1 training (Triage, Urgency, Parts)
│   ├── train_m2.py               # Module M2 training (Duration, Cost)
│   ├── train_m3.py               # Module M3 training (Lead Conversion, Pricing)
│   ├── train_m4a.py              # Module M4A training (RFM, Repeat Visit)
│   ├── train_m4b.py              # Module M4B training (Territory Demand)
│   └── evaluate_all.py           # Module M5 comprehensive evaluation synthesis
├── requirements.txt              # Pinned production dependencies
├── run_pipeline.py               # Single-command automated pipeline runner
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Shivang1001/Machine_Learning_Project.git
cd Machine_Learning_Project
```

### 2. Set Up a Virtual Environment

* **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 3. Install Dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Search Path & Verify Data
Ensure the project root is in your `PYTHONPATH`:

* **Linux / macOS:** `export PYTHONPATH=.`
* **Windows (PowerShell):** `$env:PYTHONPATH="."`

Verify data loading:
```bash
python -c "from src.io import load; df = load('jobs'); print(f'Loaded {len(df):,} job records successfully.')"
```

---

## ⚙️ Model Training Pipeline

You can train all analytical modules sequentially with a single command or run them individually.

### Option A: Automated Single-Command Execution
```bash
python run_pipeline.py
```

### Option B: Step-by-Step Modular Execution
```bash
# Step 1: Module M1 (Intake Triage, Urgency & Parts)
python src/train_m1.py

# Step 2: Module M2 (Duration & Internal Cost Estimation)
python src/train_m2.py

# Step 3: Module M3 (Conversion Scoring & Price Optimization)
python src/train_m3.py

# Step 4: Module M4A (Customer Segmentation & Repeat Propensity)
python src/train_m4a.py

# Step 5: Module M4B (Territory Demand Forecasting)
python src/train_m4b.py

# Step 6: Module M5 (Global Evaluation & Benchmark Synthesis)
python src/evaluate_all.py
```

All trained artifacts will be saved in `models/`, and diagnostic figures/metrics will be written to `reports/`.

---

## 🖥️ Launching the Operations Console

Once model training is complete, start the interactive Streamlit operations dashboard:

```bash
streamlit run app/streamlit_app.py
```

The application will launch locally at `http://localhost:8501`.

### Console Features:
* **Sidebar System Status:** Real-time readiness check for all trained `.joblib` model binaries.
* **Tab 1: Triage & Live Quotation:** Enter raw service tickets to generate real-time fault classifications, SLA urgency levels, spare parts loadout manifests, and profit-optimized quote recommendations.
* **Tab 2: Demand & Capacity Planning:** Inspect weekly technician workload projections, capacity utilization bounds, and staffing deficit alerts by territory.
* **Tab 3: Customer Retention & Cohorts:** Explore RFM customer clusters, high-risk churn indicators, and prioritized follow-up outreach lists.
* **Tab 4: Model Scoreboard:** Inspect academic validation metrics, baseline comparisons, and empirical operational lift figures.

---

## 🛠️ Tech Stack & Toolchain

* **Core Language:** Python 3.10 / 3.11
* **Machine Learning & NLP:** `scikit-learn`, `scipy`, `statsmodels`, `joblib`
* **Data Processing & Analytics:** `pandas`, `numpy`, `pyarrow`
* **Visualization & Dashboard:** `streamlit`, `plotly`, `matplotlib`

---

## 📄 License & Attribution

This project is developed as an academic and engineering capstone for intelligent field service operations. Distributed under the MIT License.
VOLTLINE PROJECT CONTRACT v1.0

You are one of six teammates building the same college ML project in parallel. Each teammate gives you this same contract with a different MY_MODULE line at the bottom. Your output must plug into the others' work with zero edits, so follow this contract exactly. It overrides your own preferences and habits. Never rename, add, remove or reorder anything it defines. If something is missing or contradictory, use the stated default, continue, and list it under CONTRACT QUESTIONS at the end of your reply. Do not ask me questions before starting, except in the situation described in rule 9.

=====================================================
1. PROJECT
=====================================================
Voltline Smart Service Operations Platform: machine learning for a multi-branch electrical service franchise (like Mr Electric). One-week group project.

Modules:
M0A  synthetic data generator (owner of all data files and src/config.py, src/io.py)
M0B  data validation and EDA
M1   job triage: fault category, urgency, parts needed (text classification)
M2   job duration and cost estimator (regression)
M3A  lead conversion scoring (classification)
M3B  quote optimiser (calculation on top of M2 and M3A)
M4A  customer segmentation and repeat-job prediction
M4B  weekly demand forecasting and staffing
M5   Streamlit app and business impact evaluation

HARD LIMIT: classical machine learning only. No neural networks or deep learning of any kind (no TensorFlow, PyTorch, Keras, transformers, LSTM, pretrained embeddings). Allowed: scikit-learn, LightGBM, statsmodels (baselines only).

All project data is synthetic and produced by M0A. Public datasets (UCI household power, UCI appliances energy, NYC 311, IBM Telco churn) are only cited as references in the report and must never be mixed into the project files.

=====================================================
2. ENVIRONMENT
=====================================================
Python 3.10+. requirements.txt (use exactly these, add nothing):
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.2
lightgbm==4.5.0
statsmodels==0.14.3
joblib==1.4.2
streamlit==1.39.0
plotly==5.24.1
Faker==30.8.0

Run everything from the repo root as: python -m src.<script>   (generator: python -m generator.generate_data)

src/config.py (owned by M0A; everyone imports from it; nobody redefines these values):
RANDOM_STATE = 42
DATA_START = "2024-01-01"; DATA_END = "2025-12-31"
TRAIN_END = "2025-06-30"; VAL_END = "2025-09-30"
FAULTS = ["wiring","mcb","ac","geyser","fan","inverter","light"]
URGENCY = ["low","medium","high"]
APPLIANCES = ["ac","geyser","fan_cooler","inverter_ups","switchboard","lights_switches","unknown"]
PREMISES = ["apartment","house","shop_office"]
CHANNELS = ["google_ads","referral","website","whatsapp","walk_in"]
BRANCHES = ["NCR","PUN","JAI"]
PARTS = ["copper_wire","connectors_tape","switch_socket","mcb","rccb","cable_lugs","run_capacitor","contactor","isolator","thermostat","heating_element","geyser_plug","fan_capacitor","regulator","bearing_set","battery_terminal","fuse","led_driver"]
JOBS_PER_TECH_PER_WEEK = 13

src/io.py (owned by M0A):
load(name) -> DataFrame; name is one of branches, technicians, customers, leads, jobs, weather, rate_card. Date columns are parsed to datetime.
split(df, date_col) -> (train, val, test) using TRAIN_END and VAL_END: train is date <= TRAIN_END, val is TRAIN_END < date <= VAL_END, test is date > VAL_END.

=====================================================
3. REPOSITORY LAYOUT AND OWNERSHIP
=====================================================
voltline/
  requirements.txt, CONTRACT.md
  data/processed/*.csv, data/processed/data_dictionary.md, data/processed/CHECKSUMS.sha256   (M0A only; frozen after Day 2)
  generator/generate_data.py                                  (M0A)
  src/config.py, src/io.py                                    (M0A)
  src/validate_data.py, src/eda.py                            (M0B)
  src/m1_triage.py, src/train_m1.py                           (M1)
  src/m2_estimator.py, src/train_m2.py                        (M2)
  src/m3a_leads.py, src/train_m3a.py                          (M3A)
  src/m3b_quote.py                                            (M3B)
  src/m4a_segments.py, src/train_m4a.py                       (M4A)
  src/m4b_demand.py, src/train_m4b.py                         (M4B)
  app/streamlit_app.py, src/evaluate_impact.py                (M5)
  models/<module>_<name>.joblib       reports/metrics/<module>.json       reports/figures/<module>_*.png
  tests/test_<module>.py, tests/fixtures/                     (fixtures: 200-row schema-valid samples, see rule 9)

Only create or edit files that belong to MY_MODULE. Never edit another module's files or src/config.py. If you need a change there, list it under CONTRACT QUESTIONS.

=====================================================
4. DATA CONTRACT (all CSV, UTF-8, header row, snake_case, ISO dates, sorted by primary key)
=====================================================
branches.csv: branch_id, city, n_technicians
  NCR Delhi NCR 9; PUN Pune 6; JAI Jaipur 5

technicians.csv: tech_id (T-001...), branch_id, experience_years (int 1-15), skill_level (int 1-3)

customers.csv: customer_id (C-00001...), branch_id, premises_type, house_age_years (int 0-50, about 2% missing), signup_date

rate_card.csv: fault_category, premises_type, base_price_inr
  typical_price = base_price_inr * (1.12 if urgency == "high" else 1.0)

leads.csv: lead_id (L-000001...), created_at (datetime), customer_id, branch_id, source_channel, complaint_text, appliance_type, premises_type, house_age_years, fault_category, urgency, response_time_min (int), typical_price, quoted_price, quote_ratio (= quoted_price / typical_price), booked (0 or 1)

jobs.csv: job_id (J-000001...), lead_id, customer_id, tech_id, branch_id, job_date, created_hour (0-23), day_of_week (0 = Monday), complaint_text, appliance_type, premises_type, house_age_years, fault_category, urgency, parts_used (part ids joined by "|", empty string if none), n_parts, duration_hours, job_cost_inr (business cost: labour + parts + visit), final_price_inr (equals quoted_price), repeat_visit (0 or 1)
  One job per booked lead.

weather.csv: date, branch_id, temp_c, rain_mm, festival_flag (0 or 1), one row per branch per day

Date column used for splitting: leads.created_at, jobs.job_date.
Splits are always time-based using src/io.split. Never shuffle across time.
Tune with cross-validation on train only (StratifiedKFold for classifiers, TimeSeriesSplit for time series, both with random_state 42 where applicable). Choose the model on val. Refit the chosen model on train + val. Report test metrics once, at the end.

Leakage rule: a model may only use features that exist at the moment it runs. Downstream modules train on ground-truth upstream labels (for example the true fault_category) and receive predicted ones at inference. State this mismatch in your summary.

Pipeline order and who supplies what:
M1 (text, appliance, premises, house age, hour) -> fault, urgency, parts
M2 (M1 outputs + technician details) -> duration, cost
M3A (cost-independent lead features + quote_ratio) -> booking probability
M3B (M2 cost + M3A probability) -> recommended price

=====================================================
5. MODULE SPECIFICATIONS
=====================================================

---- M0A  Data generator ----
File: generator/generate_data.py, with np.random.default_rng(42) used everywhere. Same seed must give byte-identical CSVs. Also write data_dictionary.md and CHECKSUMS.sha256 (sha256 of each CSV).
Targets: about 25,000 to 28,000 jobs, 65,000 to 75,000 leads, 3 branches, 20 technicians, 2024-01-01 to 2025-12-31.
Rules (implement all, with noise so models score realistically below perfect):
- Weekly job mean per branch = base x season x festival. Base: NCR 112, PUN 78, JAI 64. Season multiplier by month (Jan to Dec): 0.85, 0.85, 0.95, 1.10, 1.20, 1.15, 1.10, 1.10, 1.00, 0.95, 1.05, 0.90. Festival weeks (week containing the date) multiply by 1.2: Holi 2024-03-25, Diwali 2024-11-01, Holi 2025-03-14, Diwali 2025-10-20. Draw weekly totals from a Poisson distribution, then spread over days.
- Fault base probabilities: wiring 0.16, mcb 0.18, ac 0.17, geyser 0.12, fan 0.13, inverter 0.10, light 0.14. Multiply ac and fan by 2.2 in Apr to Jun, geyser by 2.0 in Dec to Feb, wiring and mcb by 1.6 in Jul to Sep, then renormalise.
- appliance_type follows the fault (ac->ac, geyser->geyser, fan->fan_cooler, inverter->inverter_ups, mcb and wiring->switchboard, light->lights_switches) except 15% "unknown" and 5% random.
- Premises: apartment 0.55, house 0.30, shop_office 0.15. house_age_years = clip(round(N(18,10)), 0, 50). 2% missing.
- complaint_text: at least 20 templates per fault, mixing English, Hindi written in Latin script (Hinglish) and mixed sentences; random synonym swaps; 8% of words get one character typo. Include danger phrases for urgency (spark, sparking, smoke, burning smell, jalne ki smell, shock, melting wire), power-loss phrases (light nahi hai, no power), and mild phrases (flickering, slow, noise).
- urgency: score = base by fault (wiring 2.0, mcb 1.2, ac 0.8, geyser 1.0, fan 0.3, inverter 0.8, light 0.2) + 1.5 per distinct danger phrase + 0.5 if shop_office + 0.5 if power-loss phrase - 0.8 if only mild phrases + N(0, 0.3). high if score >= 2.5, medium if >= 1.0, else low. This rule stands in for dispatcher decisions and must be stated in the report.
- duration_hours = base_hours[fault] x (1 + 0.012 x max(0, house_age - 10)) x (1.2 - 0.035 x min(experience_years, 10)) x premises_factor x lognormal(0, 0.15). base_hours: wiring 2.6, mcb 1.4, ac 2.2, geyser 1.6, fan 1.1, inverter 1.8, light 0.8. premises_factor: apartment 1.0, house 1.15, shop_office 1.25. 1% of rows get duration x 3 (outliers).
- parts_used: each part is drawn independently with a probability from a per-fault table (for example geyser: thermostat 0.6, heating_element 0.45, geyser_plug 0.3). n_parts = count.
- job_cost_inr = 380 x duration_hours + sum of part unit prices + 199. Part unit prices are fixed in the generator and documented in data_dictionary.md.
- rate_card base_price_inr = 1.55 x mean job_cost_inr for that fault and premises, rounded to the nearest 10.
- Leads: response_time_min = clip(lognormal(median 30, sigma 0.8), 2, 300). source_channel probabilities in CHANNELS order: 0.30, 0.20, 0.20, 0.20, 0.10, with booking multipliers 0.9, 1.2, 1.0, 1.1, 1.3. quote_ratio = clip(N(1.10, 0.20), 0.70, 1.80). resp_f = 0.6 + 0.4 x exp(-response_time_min / 70). p_book = clip(channel_multiplier x resp_f / (1 + exp(k x (quote_ratio - 1.08))), 0, 1), with k = 6 (4.2 if urgency is high). booked ~ Bernoulli(p_book). Overall booking rate should land between 30% and 45%.
- Jobs exist only for booked leads. job_date = lead date + 0 to 3 days (0 to 1 for high urgency). Technician is a random technician from the lead's branch.
- repeat_visit probability = 0.04 + 0.10 x (experience_years < 3) + 0.05 x (n_parts >= 3).
- About 70% of customers have exactly one job; 30% have 2 to 8. Probability of a repeat within 6 months rises with number of previous jobs and falls with days since the last job; geyser, ac and inverter jobs repeat on an annual cycle.
- weather: temp_c = branch base (NCR 25, PUN 26, JAI 26) + 10 x sin(2 pi (day_of_year - 105) / 365) + N(0, 2); rain_mm is heavy from July to September, light otherwise; festival_flag is 1 for three days around each festival date above.
Done when: all files exist, python -m src.validate_data passes, and a second run reproduces identical CHECKSUMS.

---- M0B  Validation and EDA ----
src/validate_data.py asserts: file and column names and order, dtypes, enumerations match src/config.py, primary keys unique, foreign keys resolve, date ranges, row counts inside the M0A targets, quote_ratio = quoted_price / typical_price, one job per booked lead, and that the intended signals exist (AC share higher in Apr to Jun; booking rate falls as quote_ratio rises and as response_time_min rises; duration rises with house age and falls with experience). Write the result to reports/data_validation.json.
src/eda.py writes figures to reports/figures/m0b_*.png: weekly jobs by branch, fault mix by month, urgency distribution, booking rate by quote_ratio bin, duration by fault, text length distribution, missing values.
Until M0A delivers data, write both scripts against the schema and test them on tests/fixtures.

---- M1  Job triage ----
src/m1_triage.py, src/train_m1.py. Three models on the same text features.
Text features: lowercase and strip punctuation, then union of TfidfVectorizer(ngram_range=(1,2), min_df=2) and TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3).
Features used: complaint_text, appliance_type, premises_type, house_age_years (median-imputed), created_hour. Not allowed: technician fields, response_time_min, prices, any other leads or jobs column.
Models:
- Fault (multi-class): LogisticRegression(max_iter=2000, class_weight="balanced"), C chosen from [0.5, 1, 2, 5]. Baselines: LinearSVC, MultinomialNB. Primary metric: macro F1.
- Urgency (multi-class): LogisticRegression(max_iter=2000, class_weight="balanced") on text features + structured features. Comparison: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=42) on structured features + TF-IDF word features limited to max_features=2000. Primary metric: recall of class "high". Also report macro F1 and the confusion matrix, because confusing high with low is worse than high with medium.
- Parts (multi-label over PARTS): OneVsRestClassifier(LogisticRegression(max_iter=2000, class_weight="balanced")) on text features + true fault_category one-hot at training, predicted fault at inference. Primary metric: micro F1. Also Hamming loss.
Saved files: models/m1_fault.joblib, models/m1_urgency.joblib, models/m1_parts.joblib (fitted sklearn Pipelines, or a dict of pipeline plus label list for parts).
Also write reports/figures/m1_confusion_fault.png and m1_top_words.png (top 10 TF-IDF words per fault from coefficients).

---- M2  Duration and cost estimator ----
src/m2_estimator.py, src/train_m2.py.
Features: fault_category, appliance_type, premises_type, house_age_years (median-imputed), urgency, branch_id, experience_years, skill_level, month (from job_date). Not allowed: n_parts, parts_used, final_price_inr, repeat_visit, lead columns.
Targets: duration_hours, and job_cost_inr trained as log1p(job_cost_inr) then back-transformed.
Preprocessing: ColumnTransformer with OneHotEncoder(handle_unknown="ignore") for categoricals and StandardScaler for numerics.
Models per target: Ridge (baseline), RandomForestRegressor(n_estimators=300, random_state=42), LGBMRegressor(n_estimators=400, learning_rate=0.05, random_state=42). Choose by validation MAE. Primary metric: MAE; also RMSE and R2.
Interval: two LGBMRegressor(objective="quantile", alpha=0.1 and 0.9) models for duration, saved as duration_low and duration_high.
Saved files: models/m2_duration.joblib, models/m2_duration_low.joblib, models/m2_duration_high.joblib, models/m2_cost.joblib.
Also write reports/figures/m2_feature_importance.png and m2_residuals.png.

---- M3A  Lead scoring ----
src/m3a_leads.py, src/train_m3a.py.
Rows: all leads (booked 0 or 1).
Features: source_channel, response_time_min, hour (from created_at), day_of_week, appliance_type, premises_type, quote_ratio, urgency, branch_id. Not allowed: fault_category, booked-dependent columns, job columns.
Models: LogisticRegression(class_weight="balanced", max_iter=2000) baseline, LGBMClassifier(n_estimators=400, learning_rate=0.05, class_weight="balanced", random_state=42). Do not use SMOTE.
Calibrate the chosen model with CalibratedClassifierCV(method="sigmoid", cv=5) so outputs are real probabilities; M3B depends on this.
Metrics: ROC-AUC (primary), PR-AUC, Brier score. Also write reports/figures/m3a_calibration.png and m3a_win_vs_ratio.png (win probability against quote_ratio, other features fixed at the median; the curve must fall as the ratio rises).
Saved file: models/m3a_lead.joblib.

---- M3B  Quote optimiser ----
src/m3b_quote.py. No training. Uses models/m2_cost.joblib, models/m3a_lead.joblib and data/processed/rate_card.csv.
Algorithm: cost = predicted job cost (M2). typical_price = rate_card base price x (1.12 if urgency is high else 1.0). Candidate prices from 1.00 x cost to 2.60 x cost in steps of 5 rupees. For each price, quote_ratio = price / typical_price, win_prob = M3A probability, expected_profit = win_prob x (price - cost). Recommended price = the candidate with the highest expected_profit (lowest price on a tie).
Baseline for comparison: quote = cost x 1.30. Report average expected profit per lead for both on the test period leads, and the percentage lift. Write reports/metrics/m3b.json.

---- M4A  Customer segments and repeat prediction ----
src/m4a_segments.py, src/train_m4a.py.
RFM snapshot function rfm(as_of): recency_days (days since last job before as_of), frequency (jobs before as_of), monetary (sum of final_price_inr before as_of). Use only jobs with job_date <= as_of.
Segments: StandardScaler on log1p(recency_days, frequency, monetary), then KMeans(n_clusters=4, n_init=10, random_state=42), fitted on the as_of = 2025-06-30 snapshot. Report the elbow (k = 2 to 8) and silhouette in the metrics file. Name the clusters by this fixed rule: loyal_high_value = highest frequency x monetary centroid; at_risk = highest recency among the rest; new_one_time = lowest frequency among the rest; the remaining one is regular_seasonal.
Repeat model: target repeat_6m = 1 if the customer has any job in the six months after the snapshot. Train on snapshot 2024-12-31 (target window 2025-01-01 to 2025-06-30), test on snapshot 2025-06-30 (target window 2025-07-01 to 2025-12-31). Features: recency_days, frequency, monetary, avg_gap_days (fill 365 when frequency is 1), last_fault_category, premises_type, house_age_years (median-imputed), segment. Models: LogisticRegression baseline and LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42). Metrics: ROC-AUC (primary), PR-AUC.
Saved files: models/m4a_kmeans.joblib (scaler, kmeans, name map), models/m4a_repeat.joblib. Figures: m4a_elbow.png, m4a_segments_scatter.png.

---- M4B  Weekly demand forecast ----
src/m4b_demand.py, src/train_m4b.py.
Series: weekly job counts per branch using complete Monday to Sunday weeks. First week starts 2024-01-01, last complete week starts 2025-12-22. Test = last 12 weeks (starting 2025-10-06). Validation = the 12 weeks before that. Train = the rest.
Features: lag_1, lag_2, lag_4, roll4_mean (mean of the previous 4 weeks, shifted so it never includes the current week), month, week_of_year, festival_flag (1 if the week contains a festival day), temp_mean, rain_sum (the week's own weather; at forecast time use the same weeks of the previous year as the weather assumption), branch one-hot.
Models: naive baseline (last week's count), RandomForestRegressor(n_estimators=300, random_state=42), LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42). Time-series CV: TimeSeriesSplit(n_splits=5) on train. Metrics: MAE (primary), MAPE, and improvement over naive.
forecast_demand predicts 4 weeks starting 2025-12-29 recursively (each predicted week feeds the next week's lags). Interval: lower = forecast x (1 - (0.08 + 0.02 x horizon)), upper = forecast x (1 + (0.08 + 0.02 x horizon)), horizon = 1 to 4. techs_needed = ceil(forecast / JOBS_PER_TECH_PER_WEEK).
Saved file: models/m4b_demand.joblib. Figures: m4b_forecast_<branch>.png.

---- M5  App and impact evaluation ----
app/streamlit_app.py: four tabs named exactly "Triage & Quote", "Demand", "Customers", "Models & Impact". It calls only the functions in section 6; never import a trainer. Cache models with st.cache_resource and data with st.cache_data. While a module is not yet delivered, import fails gracefully and that tab shows "Module not available yet".
Triage & Quote tab: complaint text and the inputs required by the section 6 functions; shows fault, urgency, top parts, duration with range, cost, recommended quote, win probability and the profit-versus-price curve.
Demand tab: branch selector, actuals plus 4-week forecast with interval, staffing table (techs_needed against branches.n_technicians).
Customers tab: segment table, RFM scatter, reminder list (customers with repeat_prob_6m above 0.5 and recency_days above 90, top 20, downloadable as CSV).
Models & Impact tab: reads every reports/metrics/*.json and shows them in tables.
src/evaluate_impact.py writes reports/metrics/m5.json comparing against a no-ML baseline on the test period:
 - repeat_visit rate with and without parts prediction (simulated: a job whose true parts are all in the predicted top-6 list is counted as avoiding a part-related repeat),
 - average expected profit per lead: cost x 1.30 versus M3B recommended price,
 - successful conversions in the top 30% of leads ranked by M3A score versus a random 30%,
 - technician-weeks short or idle: using M4B forecast versus a constant staffing level equal to the 2025 H1 mean.
State all simulation assumptions in the file under "assumptions".

=====================================================
6. INTERFACE CONTRACT (exact names, argument names, return shapes)
=====================================================
src/m1_triage.py
  predict_triage(text: str, appliance_type: str = "unknown", premises_type: str = "apartment", house_age_years: int = 15, hour: int = 12) -> dict
  returns {"fault": str, "fault_probs": {fault: float}, "urgency": str, "urgency_probs": {level: float}, "parts": [{"part": str, "prob": float}, ...]}
  fault_probs sums to 1; parts sorted by prob descending, top 6.

src/m2_estimator.py
  predict_job(fault_category, appliance_type, premises_type, house_age_years, urgency, branch_id, experience_years, skill_level, month) -> dict
  returns {"duration_hours": float, "duration_low": float, "duration_high": float, "cost_inr": float}

src/m3a_leads.py
  score_lead(source_channel, response_time_min, hour, day_of_week, appliance_type, premises_type, quote_ratio, urgency, branch_id) -> float   (calibrated probability of booking)

src/m3b_quote.py
  recommend_quote(fault_category, appliance_type, premises_type, house_age_years, urgency, branch_id, experience_years, skill_level, month, source_channel, response_time_min, hour, day_of_week) -> dict
  returns {"price": float, "win_prob": float, "expected_profit": float, "cost_inr": float, "typical_price": float, "curve": [{"price": float, "win_prob": float, "expected_profit": float}, ...]}

src/m4a_segments.py
  get_segments(as_of: str = "2025-06-30") -> DataFrame with columns [customer_id, recency_days, frequency, monetary, segment, repeat_prob_6m]

src/m4b_demand.py
  forecast_demand(branch_id: str, weeks: int = 4) -> DataFrame with columns [week_start, forecast, lower, upper, techs_needed]

Every function loads its model from models/ at first call (lazy), is deterministic, raises ValueError with a clear message for values outside the enumerations in src/config.py, and never reads or writes anything else.

=====================================================
7. METRICS FILE SCHEMA (reports/metrics/<module>.json, identical keys for every module)
=====================================================
{
  "module": "M1",
  "contract_version": "1.0",
  "random_state": 42,
  "models": [
    {
      "name": "fault_logreg",
      "task": "fault_classification",
      "primary_metric": {"name": "macro_f1", "value": 0.0},
      "metrics": {"accuracy": 0.0},
      "baseline": {"name": "linear_svc", "metrics": {"macro_f1": 0.0}},
      "n_train": 0, "n_val": 0, "n_test": 0,
      "features": ["complaint_text", "appliance_type"],
      "best_params": {}
    }
  ],
  "notes": "one short paragraph, including the ground-truth-versus-predicted mismatch if relevant"
}
Metric names (lowercase): macro_f1, recall_high, micro_f1, hamming_loss, mae, rmse, r2, roc_auc, pr_auc, brier, mape, silhouette, accuracy. Values are floats rounded to 4 decimals. Money in INR, durations in hours.

=====================================================
8. WORKING RULES
=====================================================
1. Seed everything with RANDOM_STATE from src/config.py. Sort data by primary key before any operation that depends on order.
2. Every training script runs from the repo root, finishes in under 5 minutes on a laptop, writes its models and metrics file, and prints the metrics JSON.
3. Every module ships tests/test_<module>.py (pytest) that checks: the interface function runs, returns the exact keys and types in section 6, probabilities sum to 1 where applicable, and invalid enum values raise ValueError.
4. Use sklearn Pipelines so preprocessing is saved with the model. Fit transformers on training data only.
5. Use the exact hyperparameters named in section 5 as starting points; you may tune only the parameters the spec says to tune, using the CV procedure in section 4.
6. Do not add features, columns, files, enums or dependencies that this contract does not list.
7. Code style: short functions, type hints on public functions, a one-line docstring on each, no notebooks (scripts only) so diffs stay readable.
8. Report numbers only from code you actually ran. If you cannot execute code in this chat, say so and do not invent metrics; give the exact commands for me to run instead.
9. If the data files you need (data/processed/*.csv) are not available in this chat, do not generate substitutes. Instead: write your code against the schema in section 4, create tests/fixtures with 200 schema-valid rows built from this contract, state clearly that results on fixtures are not meaningful, and tell me to rerun on the real files. M0A is the only module that generates data.
10. If MY_MODULE lists more than one module, complete them one at a time in the order listed, each fully, before starting the next.
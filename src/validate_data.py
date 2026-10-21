"""M0B: Validates processed data against Contract v1.0 specifications and asserts domain signals."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    FAULTS, URGENCY, APPLIANCES, PREMISES,
    CHANNELS, BRANCHES, PARTS, DATA_START, DATA_END
)
from src.io import load

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_schema():
    results = {"checks": {}, "passed": True, "errors": []}

    def record_check(name: str, passed: bool, msg: str = ""):
        results["checks"][name] = "PASSED" if passed else "FAILED"
        if not passed:
            results["passed"] = False
            results["errors"].append(f"{name}: {msg}")

    # Load datasets
    branches = load("branches")
    technicians = load("technicians")
    customers = load("customers")
    rate_card = load("rate_card")
    leads = load("leads")
    jobs = load("jobs")
    weather = load("weather")

    # 1. Row counts within target boundaries
    record_check("jobs_count_range", 25000 <= len(jobs) <= 28000, f"Got {len(jobs)} jobs (target: 25k-28k)")
    record_check("leads_count_range", 65000 <= len(leads) <= 75000, f"Got {len(leads)} leads (target: 65k-75k)")
    record_check("branches_count", len(branches) == 3, f"Got {len(branches)} branches")
    record_check("technicians_count", len(technicians) == 20, f"Got {len(technicians)} technicians")

    # 2. Primary Keys uniqueness
    record_check("pk_branches_unique", branches["branch_id"].is_unique)
    record_check("pk_techs_unique", technicians["tech_id"].is_unique)
    record_check("pk_cust_unique", customers["customer_id"].is_unique)
    record_check("pk_leads_unique", leads["lead_id"].is_unique)
    record_check("pk_jobs_unique", jobs["job_id"].is_unique)

    # 3. Foreign Keys resolution
    record_check("fk_tech_branch", technicians["branch_id"].isin(branches["branch_id"]).all())
    record_check("fk_cust_branch", customers["branch_id"].isin(branches["branch_id"]).all())
    record_check("fk_leads_branch", leads["branch_id"].isin(branches["branch_id"]).all())
    record_check("fk_leads_cust", leads["customer_id"].isin(customers["customer_id"]).all())
    record_check("fk_jobs_lead", jobs["lead_id"].isin(leads["lead_id"]).all())
    record_check("fk_jobs_cust", jobs["customer_id"].isin(customers["customer_id"]).all())
    record_check("fk_jobs_tech", jobs["tech_id"].isin(technicians["tech_id"]).all())

    # 4. Relational integrity: Exactly one job per booked lead
    booked_leads = leads[leads["booked"] == 1]["lead_id"]
    unbooked_leads = leads[leads["booked"] == 0]["lead_id"]
    record_check("one_job_per_booked_lead", set(jobs["lead_id"]) == set(booked_leads))
    record_check("no_jobs_for_unbooked_lead", len(set(jobs["lead_id"]).intersection(set(unbooked_leads))) == 0)

    # 5. Enumerations & Value Ranges
    record_check("enum_faults_leads", leads["fault_category"].isin(FAULTS).all())
    record_check("enum_faults_jobs", jobs["fault_category"].isin(FAULTS).all())
    record_check("enum_urgency_leads", leads["urgency"].isin(URGENCY).all())
    record_check("enum_urgency_jobs", jobs["urgency"].isin(URGENCY).all())
    record_check("enum_appliances_leads", leads["appliance_type"].isin(APPLIANCES).all())
    record_check("enum_appliances_jobs", jobs["appliance_type"].isin(APPLIANCES).all())
    record_check("enum_premises_cust", customers["premises_type"].isin(PREMISES).all())
    record_check("enum_premises_leads", leads["premises_type"].isin(PREMISES).all())
    record_check("enum_channels_leads", leads["source_channel"].isin(CHANNELS).all())

    # 6. Parts used verification
    all_job_parts = set([p for sublist in jobs["parts_used"].dropna().str.split("|") for p in sublist if p])
    record_check("enum_parts_used", all_job_parts.issubset(set(PARTS)))

    # 7. Quote ratio calculation assertion
    expected_ratio = leads["quoted_price"] / leads["typical_price"]
    ratio_diff = (leads["quote_ratio"] - expected_ratio).abs().max()
    record_check("quote_ratio_matches_prices", ratio_diff < 0.005, f"Max diff: {ratio_diff}")

    # 8. Date ranges assertion
    min_lead = pd.to_datetime(leads["created_at"]).min()
    max_lead = pd.to_datetime(leads["created_at"]).max()
    min_job = pd.to_datetime(jobs["job_date"]).min()
    max_job = pd.to_datetime(jobs["job_date"]).max()

    leads_valid = bool((min_lead >= pd.to_datetime(DATA_START)) and (max_lead <= pd.to_datetime(DATA_END + " 23:59:59")))
    jobs_valid = bool((min_job >= pd.to_datetime(DATA_START)) and (max_job <= pd.to_datetime(DATA_END)))

    record_check("lead_dates_valid", leads_valid, f"Range: {min_lead} to {max_lead}")
    record_check("job_dates_valid", jobs_valid, f"Range: {min_job} to {max_job}")
    
    # 9. Domain Signal Existence Checks
    # Signal A: AC fault share is higher in peak summer (Apr-Jun) than winter (Dec-Feb)
    leads_month = leads["created_at"].dt.month
    summer_ac_share = (leads[leads_month.isin([4, 5, 6])]["fault_category"] == "ac").mean()
    winter_ac_share = (leads[leads_month.isin([12, 1, 2])]["fault_category"] == "ac").mean()
    record_check("signal_ac_seasonality", summer_ac_share > winter_ac_share * 1.5,
                 f"Summer AC: {summer_ac_share:.2%}, Winter AC: {winter_ac_share:.2%}")

    # Signal B: Booking rate falls as quote_ratio increases
    leads["ratio_bin"] = pd.qcut(leads["quote_ratio"], q=4)
    booking_by_ratio = leads.groupby("ratio_bin", observed=False)["booked"].mean().tolist()
    record_check("signal_price_elasticity", booking_by_ratio[0] > booking_by_ratio[-1],
                 f"Lowest ratio win: {booking_by_ratio[0]:.2%}, Highest ratio win: {booking_by_ratio[-1]:.2%}")

    # Signal C: Booking rate falls as response_time increases
    leads["resp_bin"] = pd.qcut(leads["response_time_min"], q=4)
    booking_by_resp = leads.groupby("resp_bin", observed=False)["booked"].mean().tolist()
    record_check("signal_response_time_impact", booking_by_resp[0] > booking_by_resp[-1],
                 f"Fast resp win: {booking_by_resp[0]:.2%}, Slow resp win: {booking_by_resp[-1]:.2%}")

    # Signal D: Duration rises with house age and falls with experience
    jobs_merged = jobs.merge(technicians[["tech_id", "experience_years"]], on="tech_id", suffixes=("", "_tech"))
    high_age_dur = jobs_merged[jobs_merged["house_age_years"] > 25]["duration_hours"].mean()
    low_age_dur = jobs_merged[jobs_merged["house_age_years"] <= 10]["duration_hours"].mean()
    high_exp_dur = jobs_merged[jobs_merged["experience_years"] >= 8]["duration_hours"].mean()
    low_exp_dur = jobs_merged[jobs_merged["experience_years"] <= 3]["duration_hours"].mean()
    record_check("signal_duration_vs_age", high_age_dur > low_age_dur, f"Old: {high_age_dur:.2f}h, New: {low_age_dur:.2f}h")
    record_check("signal_duration_vs_exp", low_exp_dur > high_exp_dur, f"Junior: {low_exp_dur:.2f}h, Senior: {high_exp_dur:.2f}h")

    # Output to reports/data_validation.json
    out_file = REPORTS_DIR / "data_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    if not results["passed"]:
        raise ValueError(f"Data validation failed with {len(results['errors'])} error(s).")
    print("\nAll data validation and contract assertions PASSED.")


if __name__ == "__main__":
    validate_schema()
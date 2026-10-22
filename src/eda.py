"""M0B: Exploratory Data Analysis & Diagnostic Figure Generation."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.io import load

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def generate_eda_figures():
    """Generates all contract-mandated EDA and diagnostic plots."""
    jobs = load("jobs")
    leads = load("leads")
    weather = load("weather")

    # 1. Weekly Jobs by Branch
    jobs["week_start"] = jobs["job_date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
    weekly_branch = jobs.groupby(["week_start", "branch_id"]).size().reset_index(name="job_count")
    
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=weekly_branch, x="week_start", y="job_count", hue="branch_id", palette="tab10")
    plt.title("M0B: Weekly Completed Jobs by Branch")
    plt.xlabel("Week Start Date")
    plt.ylabel("Job Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m0b_weekly_jobs_branch.png", dpi=150)
    plt.close()

    # 2. Monthly Fault Distribution
    jobs["month"] = pd.to_datetime(jobs["job_date"]).dt.to_period("M").astype(str)
    fault_month = jobs.groupby(["month", "fault_category"]).size().reset_index(name="count")

    plt.figure(figsize=(11, 4.5))
    sns.barplot(data=fault_month, x="month", y="count", hue="fault_category", palette="Set2")
    plt.title("M0B: Monthly Fault Category Distribution")
    plt.xlabel("Month")
    plt.ylabel("Job Count")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m0b_fault_distribution.png", dpi=150)
    plt.close()

    # 3. Urgency Breakdown & Price vs Duration Scatter
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=jobs.sample(min(2000, len(jobs)), random_state=42), x="duration_hours", y="job_cost_inr", hue="urgency", alpha=0.6, palette="coolwarm")
    plt.title("M0B: Job Cost vs. Duration by Urgency")
    plt.xlabel("Duration (Hours)")
    plt.ylabel("Job Cost (INR)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "m0b_cost_vs_duration.png", dpi=150)
    plt.close()

    print("[M0B EDA] Successfully generated all exploratory figures in reports/figures/")


if __name__ == "__main__":
    generate_eda_figures()
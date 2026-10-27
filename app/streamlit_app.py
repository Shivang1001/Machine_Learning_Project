"""
M5: Voltline Smart Service Operations Console.
Executive Decision Support & Real-Time Dispatch System.
Department of Computer Science & Engineering, IIT Roorkee.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import (
    APPLIANCES,
    BRANCHES,
    CHANNELS,
    FAULTS,
    JOBS_PER_TECH_PER_WEEK,
    PARTS,
    PREMISES,
    URGENCY,
)
from src.io import load

st.set_page_config(
    page_title="Voltline Ops Console | IIT Roorkee M.Tech Project",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0F172A;
}

code, .mono {
    font-family: 'JetBrains+Mono', monospace !important;
}

/* Institutional Academic Header */
.inst-header {
    background: linear-gradient(135deg, #0B192C 0%, #1E293B 70%, #1E3A8A 100%);
    border-left: 6px solid #2563EB;
    border-radius: 6px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
}
.inst-header-badge {
    font-family: 'JetBrains+Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #60A5FA;
    margin-bottom: 6px;
}
.inst-header h1 {
    font-size: 24px;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: 0.01em;
    color: #FFFFFF;
}
.inst-header p {
    font-size: 13.5px;
    margin: 0;
    color: #CBD5E1;
}

/* Operational Cards */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 14px;
    margin-bottom: 16px;
}
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #334155;
    border-radius: 4px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}
.kpi-card.highlight {
    border-top-color: #2563EB;
    background: #F8FAFC;
}
.kpi-label {
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 600;
    color: #64748B;
    margin-bottom: 4px;
    letter-spacing: 0.05em;
}
.kpi-value {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
}
.kpi-meta {
    font-size: 11.5px;
    color: #64748B;
    margin-top: 3px;
    font-family: 'JetBrains+Mono', monospace;
}

/* Status Badges */
.badge-status {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    display: inline-block;
}
.status-online { background: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
.status-pending { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def get_dataset(name: str) -> pd.DataFrame:
    return load(name)


@st.cache_data
def load_all_metrics() -> Dict[str, Dict[str, Any]]:
    metrics_dir = ROOT_DIR / "reports" / "metrics"
    out: Dict[str, Dict[str, Any]] = {}
    if metrics_dir.exists():
        for p in sorted(metrics_dir.glob("*.json")):
            try:
                out[p.stem.upper()] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return out


@st.cache_resource
def get_system_telemetry() -> Dict[str, bool]:
    m_dir = ROOT_DIR / "models"
    return {
        "M1 (NLP Triage)": (m_dir / "m1_fault.joblib").exists(),
        "M2 (Duration & Cost)": (m_dir / "m2_duration.joblib").exists() and (m_dir / "m2_cost.joblib").exists(),
        "M3A (Conversion Scoring)": (m_dir / "m3a_lead.joblib").exists(),
        "M3B (Pricing Optimization)": (ROOT_DIR / "src" / "m3b_quote.py").exists(),
        "M4A (Customer Segmentation)": (m_dir / "m4a_kmeans.joblib").exists() and (m_dir / "m4a_repeat.joblib").exists(),
        "M4B (Workforce Forecast)": (m_dir / "m4b_demand.joblib").exists(),
    }


def execute_triage(text: str, app: str, prem: str, age: int, hour: int) -> Dict[str, Any]:
    from src.m1_triage import predict_triage
    return predict_triage(text, app, prem, age, hour)


def execute_estimation(fault: str, app: str, prem: str, age: int, urg: str, branch: str, exp: int, skill: int, month: int) -> Dict[str, float]:
    from src.m2_estimator import predict_job
    return predict_job(fault, app, prem, age, urg, branch, exp, skill, month)


def execute_quote_opt(fault: str, app: str, prem: str, age: int, urg: str, branch: str, exp: int, skill: int, month: int, channel: str, resp: int, hour: int, dow: int) -> Dict[str, Any]:
    from src.m3b_quote import recommend_quote
    return recommend_quote(fault, app, prem, age, urg, branch, exp, skill, month, channel, resp, hour, dow)


def execute_segmentation(as_of: str) -> pd.DataFrame:
    from src.m4a_segments import get_segments
    return get_segments(as_of=as_of)


def execute_demand_forecast(branch: str, horizon: int) -> pd.DataFrame:
    from src.m4b_demand import forecast_demand
    return forecast_demand(branch_id=branch, weeks=horizon)


# --- SIDEBAR CONTROL PLANE ---
with st.sidebar:
    st.markdown("### Operational Settings")
    sel_branch = st.selectbox("Dispatch Branch", options=BRANCHES, index=0)
    eval_as_of = st.selectbox("Customer Snapshot Cutoff", options=["2025-06-30", "2024-12-31", "2025-09-30", "2025-12-31"], index=0)

    st.markdown("---")
    st.markdown("### Pipeline Status")
    telemetry = get_system_telemetry()
    ready_count = sum(1 for v in telemetry.values() if v)
    st.caption(f"Modules Verified: {ready_count}/{len(telemetry)}")

    for mod, is_ready in telemetry.items():
        css_cls = "status-online" if is_ready else "status-pending"
        lbl = "READY" if is_ready else "MISSING"
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; margin-bottom:6px;'>"
            f"<span style='font-size:12px;'>{mod}</span>"
            f"<span class='badge-status {css_cls}'>{lbl}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# --- INSTITUTIONAL TOP BANNER ---
st.markdown(
    """
    <div class="inst-header">
        <div class="inst-header-badge">Department of Computer Science & Engineering · IIT Roorkee</div>
        <h1>VOLTLINE SMART SERVICE OPERATIONS PLATFORM</h1>
        <p>Enterprise Field Triage, Multi-Task Diagnostics, Quantile Cost Estimation & Parametric Revenue Optimization</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_triage, tab_demand, tab_customers, tab_models = st.tabs(
    ["1. Triage & Dynamic Quote", "2. Demand & Capacity", "3. Customer Segments", "4. Model Scoreboard & Evaluation"]
)


# =========================================================================
# TAB 1: TRIAGE & QUOTE
# =========================================================================
with tab_triage:
    st.markdown("### Operational Intake Form")

    with st.form("triage_form"):
        st.markdown("#### Section A: Complaint Details")
        complaint_text = st.text_area(
            "Customer Narrative / Ticket Log",
            value="Main circuit breaker trips repeatedly with burning smell in switchboard.",
            height=90,
            help="Raw problem description reported by the client or call-center operator.",
        )

        st.markdown("#### Section B: Property & Environmental Factors")
        c1, c2, c3 = st.columns(3)
        with c1:
            appliance_type = st.selectbox("Appliance Subsystem", options=APPLIANCES, index=0)
            premises_type = st.selectbox("Premises Structure", options=PREMISES, index=0)
        with c2:
            house_age = st.number_input("Property Age (Years)", min_value=0, max_value=60, value=15)
            month = st.selectbox("Service Month", options=list(range(1, 13)), index=4)
        with c3:
            hour_of_day = st.number_input("Inquiry Hour (0-23)", min_value=0, max_value=23, value=14)
            day_of_week = st.selectbox(
                "Day of Week",
                options=[(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"), (4, "Friday"), (5, "Saturday"), (6, "Sunday")],
                format_func=lambda x: x[1],
                index=1
            )[0]

        st.markdown("#### Section C: Field Dispatch & Technician Assignment")
        c4, c5, c6 = st.columns(3)
        with c4:
            technician_exp = st.number_input("Technician Experience (Years)", min_value=1, max_value=20, value=5)
        with c5:
            skill_tier = st.selectbox("Technician Skill Tier", options=[1, 2, 3], index=1, help="Tier 1: Junior, Tier 2: Standard, Tier 3: Master/Specialist")
        with c6:
            source_channel = st.selectbox("Lead Source Channel", options=CHANNELS, index=0)
            response_time = st.slider("Response Latency (Minutes)", min_value=2, max_value=120, value=20, help="Time elapsed between inquiry receipt and operational response.")

        submit_btn = st.form_submit_button("Run Operational Inference & Optimize Quote", type="primary", use_container_width=True)

    if submit_btn or "last_m1_result" in st.session_state:
        if submit_btn:
            try:
                st.session_state.last_m1_result = execute_triage(
                    complaint_text, appliance_type, premises_type, int(house_age), int(hour_of_day)
                )
                st.session_state.last_m2_result = execute_estimation(
                    st.session_state.last_m1_result["fault"], appliance_type, premises_type,
                    int(house_age), st.session_state.last_m1_result["urgency"], sel_branch,
                    int(technician_exp), int(skill_tier), int(month)
                )
                st.session_state.last_m3b_result = execute_quote_opt(
                    st.session_state.last_m1_result["fault"], appliance_type, premises_type,
                    int(house_age), st.session_state.last_m1_result["urgency"], sel_branch,
                    int(technician_exp), int(skill_tier), int(month),
                    source_channel, int(response_time), int(hour_of_day), int(day_of_week)
                )
            except Exception as e:
                st.error(f"Inference execution failed: {e}")
                st.stop()

        m1_res = st.session_state.last_m1_result
        m2_res = st.session_state.last_m2_result
        m3b_res = st.session_state.last_m3b_result

        st.markdown("---")
        st.markdown("### Operational Decisions & Predictive Metrics")

        fault_label = m1_res["fault"].upper()
        urg_label = m1_res["urgency"].upper()
        conf = m1_res["fault_probs"][m1_res["fault"]]

        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card highlight">
                    <div class="kpi-label">Classified Fault (M1)</div>
                    <div class="kpi-value">{fault_label}</div>
                    <div class="kpi-meta">Confidence: {conf:.2%}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Dispatch Urgency (M1)</div>
                    <div class="kpi-value">{urg_label}</div>
                    <div class="kpi-meta">SLA Target: {'< 45m' if urg_label == 'HIGH' else '< 4h'}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Estimated Duration (M2)</div>
                    <div class="kpi-value">{m2_res['duration_hours']:.1f} hrs</div>
                    <div class="kpi-meta">P10-P90: [{m2_res['duration_low']:.1f}h - {m2_res['duration_high']:.1f}h]</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Internal Business Cost (M2)</div>
                    <div class="kpi-value">INR {m2_res['cost_inr']:,.0f}</div>
                    <div class="kpi-meta">Labor + Expected Parts</div>
                </div>
                <div class="kpi-card highlight">
                    <div class="kpi-label">Recommended Quote (M3B)</div>
                    <div class="kpi-value" style="color:#2563EB;">INR {m3b_res['price']:,.0f}</div>
                    <div class="kpi-meta">Est. Win Probability: {m3b_res['win_prob']:.1%}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Vehicle Spare Parts Allocation Manifest")
        st.caption("Inventory loadout criteria: Parts with posterior probability ≥ 0.35 must be stocked in the service van.")
        parts_data = []
        for p in m1_res["parts"]:
            parts_data.append({
                "Part Name": p["part"],
                "Posterior Probability": f"{p['prob']:.2%}",
                "Dispatch Decision": "LOAD ON VEHICLE" if p["prob"] >= 0.35 else "DO NOT LOAD",
            })
        parts_df = pd.DataFrame(parts_data)
        st.dataframe(parts_df, use_container_width=True, hide_index=True)

        st.markdown("#### Dynamic Price Optimization & Demand Elasticity")
        st.caption("Evaluates win probability against internal business costs to find the profit-maximizing quote price.")

        curve_df = pd.DataFrame(m3b_res["curve"])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Primary axis: Expected Profit (Vibrant Cyan / Sky Blue)
        fig.add_trace(
            go.Scatter(
                x=curve_df["price"],
                y=curve_df["expected_profit"],
                name="Expected Profit (INR)",
                mode="lines+markers",
                marker=dict(size=4, color="#38BDF8"),
                line=dict(color="#38BDF8", width=3),
            ),
            secondary_y=False,
        )
        
        # Secondary axis: Conversion Win Probability (Bright Coral / Amber, Dashed)
        fig.add_trace(
            go.Scatter(
                x=curve_df["price"],
                y=curve_df["win_prob"] * 100,
                name="Win Probability (%)",
                mode="lines",
                line=dict(color="#FB923C", width=2.5, dash="dash"),
            ),
            secondary_y=True,
        )
        
        # Optimal Vertical Marker (Bright Emerald / Lime Green)
        fig.add_vline(
            x=m3b_res["price"],
            line_dash="solid",
            line_color="#4ADE80",
            line_width=2.5,
            annotation_text=f"Optimal: INR {m3b_res['price']:.0f}",
            annotation_position="top left",
            annotation_font=dict(color="#4ADE80", size=12),
        )
        
        fig.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(title_text="Quoted Price (INR)", showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
        fig.update_yaxes(title_text="Expected Profit (INR)", secondary_y=False, showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
        fig.update_yaxes(title_text="Win Probability (%)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Pricing vs. Win Probability Benchmarks")
        st.caption("Pricing and expected profit sampled at discrete win probability thresholds:")

        target_probs = [1.00, 0.75, 0.50, 0.25, 0.10]
        rows = []
        for tp in target_probs:
            # Find the row in curve_df closest to this target probability
            idx = (curve_df["win_prob"] - tp).abs().idxmin()
            match = curve_df.loc[idx]
            rows.append({
                "Target Win Rate": f"{int(tp * 100)}%",
                "Quoted Price": f"INR {match['price']:,.0f}",
                "Model Win Probability": f"{match['win_prob'] * 100:.1f}%",
                "Expected Profit": f"INR {match['expected_profit']:,.2f}",
            })

        prob_table_df = pd.DataFrame(rows).drop_duplicates(subset=["Target Win Rate"])
        st.dataframe(prob_table_df, use_container_width=True, hide_index=True)


# =========================================================================
# TAB 2: DEMAND & CAPACITY
# =========================================================================
with tab_demand:
    st.markdown(f"### Workforce Demand Forecast & Capacity Allocation: Branch {sel_branch}")

    horizon = st.slider("Forecast Horizon (Weeks)", min_value=1, max_value=8, value=4)

    try:
        df_forecast = execute_demand_forecast(sel_branch, horizon)
        branches_df = get_dataset("branches")
        staff_base = int(branches_df.loc[branches_df["branch_id"] == sel_branch, "n_technicians"].iloc[0])
        staff_cap = staff_base * JOBS_PER_TECH_PER_WEEK

        fig_dem = go.Figure()
        fig_dem.add_trace(
            go.Scatter(
                x=pd.concat([df_forecast["week_start"], df_forecast["week_start"][::-1]]),
                y=pd.concat([df_forecast["upper"], df_forecast["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                name="90% Confidence Interval",
            )
        )
        fig_dem.add_trace(
            go.Scatter(
                x=df_forecast["week_start"],
                y=df_forecast["forecast"],
                mode="lines+markers",
                line=dict(color="#38BDF8", width=2.5),
                name="Forecast Demand",
            )
        )
        fig_dem.add_hline(
            y=staff_cap, line_dash="dash", line_color="#EF4444",
            annotation_text=f"Nominal Capacity ({staff_cap} Jobs)",
            annotation_position="top left",
        )
        fig_dem.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_dem.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
        fig_dem.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
        st.plotly_chart(fig_dem, use_container_width=True)

        st.markdown("#### Weekly Staffing Balance Schedule")
        disp_df = df_forecast.copy()
        disp_df["Allocated Technicians"] = staff_base
        disp_df["Capacity Gap (Techs)"] = disp_df["techs_needed"] - staff_base
        st.dataframe(
            disp_df[["week_start", "forecast", "techs_needed", "Allocated Technicians", "Capacity Gap (Techs)"]].rename(
                columns={"week_start": "Week Commencing", "forecast": "Expected Job Count", "techs_needed": "Required Technicians"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"Demand forecasting error: {exc}")


# =========================================================================
# TAB 3: CUSTOMER SEGMENTS (M4A)
# =========================================================================
with tab_customers:
    st.markdown(f"### Customer Cohort Segmentation & Retention (Snapshot: {eval_as_of})")

    try:
        df_seg = execute_segmentation(as_of=eval_as_of)

        summary_df = (
            df_seg.groupby("segment")
            .agg(
                Customer_Count=("customer_id", "count"),
                Mean_Recency=("recency_days", "mean"),
                Mean_Frequency=("frequency", "mean"),
                Mean_Monetary=("monetary", "mean"),
                Mean_Repeat_Prob=("repeat_prob_6m", "mean"),
            )
            .reset_index()
        )

        st.markdown("#### Portfolio Cluster Distribution")
        st.dataframe(
            summary_df.style.format({
                "Mean_Recency": "{:.1f} days",
                "Mean_Frequency": "{:.2f}",
                "Mean_Monetary": "INR {:,.0f}",
                "Mean_Repeat_Prob": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Behavioral Feature Distribution (Monetary vs. Recency)")
        sample_seg = df_seg.sample(min(1000, len(df_seg)), random_state=42)
        fig_cluster = px.scatter(
            sample_seg,
            x="recency_days",
            y="monetary",
            color="segment",
            size="frequency",
            labels={"recency_days": "Recency (Days)", "monetary": "Monetary Value (INR)"},
            color_discrete_sequence=["#38BDF8", "#818CF8", "#34D399", "#FB923C"],
        )
        fig_cluster.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_cluster.update_xaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
        fig_cluster.update_yaxes(showgrid=True, gridcolor="rgba(148, 163, 184, 0.2)")
        st.plotly_chart(fig_cluster, use_container_width=True)

        st.markdown("#### High-Priority Retention Outreach Queue")
        st.caption("Target: Customers with ≥ 50% repeat likelihood who have not booked service in over 90 days.")
        reminder_df = (
            df_seg[(df_seg["repeat_prob_6m"] >= 0.50) & (df_seg["recency_days"] > 90)]
            .sort_values("repeat_prob_6m", ascending=False)
            .head(15)
        )
        st.dataframe(
            reminder_df[["customer_id", "recency_days", "monetary", "repeat_prob_6m"]].rename(
                columns={
                    "customer_id": "Customer ID",
                    "recency_days": "Recency (Days)",
                    "monetary": "Historical Spend (INR)",
                    "repeat_prob_6m": "Repeat Propensity",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.error(f"Customer segmentation pipeline execution failed: {exc}")


# =========================================================================
# TAB 4: MODELS & EVALUATION
# =========================================================================
with tab_models:
    st.markdown("### Validated Model Performance Scoreboard")

    all_metrics = load_all_metrics()
    if all_metrics:
        score_rows = []
        for m_name, blob in all_metrics.items():
            for m in blob.get("models", []):
                base_info = m.get("baseline", {})
                base_str = "None"
                if base_info.get("metrics"):
                    metric_key = list(base_info["metrics"].keys())[0]
                    base_str = f"{base_info.get('name', 'base')} ({metric_key}: {base_info['metrics'][metric_key]:.4f})"

                score_rows.append({
                    "Module": blob.get("module", m_name),
                    "Model Name": m.get("name"),
                    "Task Type": m.get("task"),
                    "Primary Test Metric": f"{m.get('primary_metric', {}).get('name')}: {m.get('primary_metric', {}).get('value', 0):.4f}",
                    "Baseline Reference": base_str,
                    "Train / Val / Test Observations": f"{m.get('n_train', 0)} / {m.get('n_val', 0)} / {m.get('n_test', 0)}",
                })
        if score_rows:
            st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Empirical Impact Summary (vs. Non-ML Operations)")
    if "M5" in all_metrics and "impact_summary" in all_metrics["M5"]:
        imp = all_metrics["M5"]["impact_summary"]
        st.markdown(
            f"""
            <div class="kpi-container">
                <div class="kpi-card">
                    <div class="kpi-label">Return Visit Reduction</div>
                    <div class="kpi-value">{imp.get('repeat_rate_baseline', 0)*100:.1f}% → {imp.get('repeat_rate_ml', 0)*100:.1f}%</div>
                    <div class="kpi-meta">Parts Coverage: {imp.get('parts_coverage_rate', 0)*100:.0f}%</div>
                </div>
                <div class="kpi-card highlight">
                    <div class="kpi-label">Profit Lift per Lead</div>
                    <div class="kpi-value">+{imp.get('profit_lift_pct', 0):.1f}%</div>
                    <div class="kpi-meta">INR {imp.get('profit_per_lead_baseline', 0):.0f} → INR {imp.get('profit_per_lead_ml', 0):.0f}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Top 30% Lead Conversion Lift</div>
                    <div class="kpi-value">+{imp.get('top30_conv_lift_pct', 0):.1f}%</div>
                    <div class="kpi-meta">ML Prioritization vs. Random</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Staffing Mismatch Reduction</div>
                    <div class="kpi-value">{imp.get('tech_weeks_mismatch_baseline', 0)} → {imp.get('tech_weeks_mismatch_ml', 0)} wks</div>
                    <div class="kpi-meta">Misallocated Tech-Weeks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Diagnostic Figure Gallery")
    fig_dir = ROOT_DIR / "reports" / "figures"
    png_list = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if png_list:
        for path in png_list:
            st.image(str(path), caption=path.name, use_column_width=True)
    else:
        st.info("No diagnostic plots found in reports/figures/.")
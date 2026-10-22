"""M5: Voltline Smart Service Operations Console — Executive & Dispatcher ML Interface."""

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure repo root is on sys.path when invoked via `streamlit run app/streamlit_app.py`
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

# -------------------------------------------------------------------------
# PAGE CONFIGURATION & INDUSTRIAL CONTROL-ROOM STYLING
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Voltline Ops Console | ML Decision Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --copper: #E3804B;
    --copper-dark: #B5501F;
    --ph-red: #EF4444;
    --ph-yellow: #F59E0B;
    --ph-blue: #3B82F6;
    --ok-green: #10B981;
    --surface-card: rgba(255, 255, 255, 0.03);
    --border-line: rgba(148, 163, 184, 0.22);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1, h2, h3, .console-title {
    font-family: 'Barlow Semi Condensed', sans-serif !important;
    letter-spacing: 0.03em;
}

.mono-num {
    font-family: 'IBM Plex Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}

/* Header Banner */
.ops-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #2A1810 100%);
    border: 1px solid rgba(227, 128, 75, 0.35);
    border-left: 6px solid #E3804B;
    border-radius: 8px;
    padding: 18px 24px;
    margin-bottom: 18px;
    color: #F8FAFC;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}
.ops-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #FFFFFF;
}
.ops-header h1 span {
    color: #E3804B;
}
.ops-header p {
    margin: 4px 0 0 0;
    font-size: 13.5px;
    color: #94A3B8;
}
.telemetry-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(227, 128, 75, 0.4);
    color: #E2E8F0;
    padding: 6px 12px;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* KPI Strip Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
}
.kpi-card {
    background: var(--surface-card);
    border: 1px solid var(--border-line);
    border-top: 3px solid #E3804B;
    border-radius: 6px;
    padding: 14px 16px;
}
.kpi-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    opacity: 0.72;
    margin-bottom: 4px;
}
.kpi-val {
    font-family: 'Barlow Semi Condensed', sans-serif;
    font-size: 27px;
    font-weight: 700;
    line-height: 1.15;
}
.kpi-sub {
    font-size: 12px;
    opacity: 0.75;
    margin-top: 4px;
}

/* Phase Urgency Pills */
.phase-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 3px 10px;
    border-radius: 4px;
    border: 1px solid currentColor;
}
.phase-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    display: inline-block;
}
.u-high { color: #EF4444; background: rgba(239, 68, 68, 0.12); }
.u-medium { color: #F59E0B; background: rgba(245, 158, 11, 0.12); }
.u-low { color: #3B82F6; background: rgba(59, 130, 246, 0.12); }

/* Parts Manifest Chips */
.parts-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
}
.part-chip {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 13px;
    padding: 6px 12px;
    border-radius: 5px;
    border: 1px solid var(--border-line);
    background: var(--surface-card);
}
.part-chip.carry {
    border: 1.5px solid #E3804B;
    background: rgba(227, 128, 75, 0.15);
    font-weight: 600;
}
.part-chip span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11.5px;
    opacity: 0.85;
    margin-left: 6px;
}

/* XAI Token Highlight Box */
.xai-box {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    line-height: 1.7;
    padding: 12px 14px;
    border-radius: 6px;
    border: 1px solid var(--border-line);
    background: var(--surface-card);
    margin-top: 8px;
}
.tok-danger {
    background: rgba(239, 68, 68, 0.25);
    color: #FCA5A5;
    border-bottom: 2px solid #EF4444;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 600;
}
.tok-power {
    background: rgba(245, 158, 11, 0.25);
    color: #FCD34D;
    border-bottom: 2px solid #F59E0B;
    padding: 1px 5px;
    border-radius: 3px;
    font-weight: 600;
}
.tok-mild {
    background: rgba(59, 130, 246, 0.25);
    color: #93C5FD;
    border-bottom: 2px solid #3B82F6;
    padding: 1px 5px;
    border-radius: 3px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------------------
# CACHED DATA & RESOURCE LOADERS
# -------------------------------------------------------------------------
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
def get_module_status() -> Dict[str, bool]:
    m_dir = ROOT_DIR / "models"
    return {
        "M1 (Triage)": (m_dir / "m1_fault.joblib").exists(),
        "M2 (Estimator)": (m_dir / "m2_duration.joblib").exists(),
        "M3A (Leads)": (m_dir / "m3a_lead.joblib").exists(),
        "M3B (Quote)": (ROOT_DIR / "src" / "m3b_quote.py").exists(),
        "M4A (Segments)": (m_dir / "m4a_kmeans.joblib").exists(),
        "M4B (Demand)": (m_dir / "m4b_demand.joblib").exists(),
    }


@st.cache_resource
def load_m1_fn():
    from src.m1_triage import predict_triage
    return predict_triage


@st.cache_resource
def load_m2_fn():
    from src.m2_estimator import predict_job
    return predict_job


@st.cache_resource
def load_m3a_fn():
    from src.m3a_leads import score_lead
    return score_lead


@st.cache_resource
def load_m3b_fn():
    from src.m3b_quote import recommend_quote
    return recommend_quote


@st.cache_resource
def load_m4a_fn():
    from src.m4a_segments import get_segments
    return get_segments


@st.cache_resource
def load_m4b_fn():
    from src.m4b_demand import forecast_demand
    return forecast_demand


# -------------------------------------------------------------------------
# DOMAIN HELPERS & EXPLAINABLE AI (XAI) UTILITIES
# -------------------------------------------------------------------------
FAULT_LABELS = {
    "wiring": "Wiring / Short Circuit",
    "mcb": "MCB / Distribution Board",
    "ac": "AC Electrical Fault",
    "geyser": "Geyser / Water Heater",
    "fan": "Fan / Cooler Motor",
    "inverter": "Inverter / UPS Backup",
    "light": "Switches & Lighting",
}

PART_PRICES_INR = {
    "copper_wire": 420, "connectors_tape": 90, "switch_socket": 180, "mcb": 380,
    "rccb": 1100, "cable_lugs": 60, "run_capacitor": 520, "contactor": 650,
    "isolator": 300, "thermostat": 350, "heating_element": 700, "geyser_plug": 150,
    "fan_capacitor": 60, "regulator": 220, "bearing_set": 180, "battery_terminal": 200,
    "fuse": 50, "led_driver": 250,
}

SAMPLE_COMPLAINTS = [
    ("⚡ Sparking & Smoke (Wiring)", "Main wiring me sparking ho rahi hai and burning smell aa rahi hai", "switchboard", "apartment"),
    ("❄️ AC Tripping MCB", "Outdoor unit making heavy humming noise and MCB trips immediately", "ac", "house"),
    ("🔥 Geyser Plug Melted", "Geyser switch board sparked and plug melted completely, garam pani nahi aa raha", "geyser", "apartment"),
    ("🔋 Inverter No Backup", "Inverter continuous long beep de raha hai aur home light nahi hai during cut", "inverter_ups", "shop_office"),
    ("🌀 Fan Slow & Humming", "Ceiling fan moving very slow even at highest regulator speed 5 and making noise", "fan_cooler", "apartment"),
]

DANGER_PHRASES = ["burning smell", "jalne ki smell", "melting wire", "sparking", "spark", "smoke", "dhuan", "shock", "melted", "burnt"]
POWER_LOSS_PHRASES = ["light nahi hai", "no power", "blackout", "completely dead"]
MILD_PHRASES = ["flickering", "slow", "noise", "dim", "humming"]


def highlight_xai_tokens(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    found_danger, found_power, found_mild = [], [], []
    annotated = text
    for p in DANGER_PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE):
            found_danger.append(p)
            annotated = re.sub(rf"(\b{re.escape(p)}\b)", r'<span class="tok-danger">🚨 \1</span>', annotated, flags=re.IGNORECASE)
    for p in POWER_LOSS_PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE):
            found_power.append(p)
            annotated = re.sub(rf"(\b{re.escape(p)}\b)", r'<span class="tok-power">⚡ \1</span>', annotated, flags=re.IGNORECASE)
    for p in MILD_PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE):
            found_mild.append(p)
            annotated = re.sub(rf"(\b{re.escape(p)}\b)", r'<span class="tok-mild">ℹ️ \1</span>', annotated, flags=re.IGNORECASE)
    return annotated, found_danger, found_power, found_mild


# -------------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ VOLTLINE TELEMETRY")
    st.caption("Multi-Branch Franchise ML Control Plane")

    mod_status = get_module_status()
    online_count = sum(1 for v in mod_status.values() if v)
    st.progress(online_count / len(mod_status), text=f"Pipeline Modules Ready: {online_count}/{len(mod_status)}")

    for m_name, is_ready in mod_status.items():
        badge = "🟢 ONLINE" if is_ready else "🟡 PENDING"
        st.markdown(f"- **{m_name}**: `{badge}`")

    st.divider()
    try:
        leads_df_meta = get_dataset("leads")
        jobs_df_meta = get_dataset("jobs")
        st.markdown("### 📊 FROZEN DATASET (M0A)")
        c1, c2 = st.columns(2)
        c1.metric("Total Leads", f"{len(leads_df_meta):,}")
        c2.metric("Booked Jobs", f"{len(jobs_df_meta):,}")
        st.caption("Horizon: `2024-01-01` → `2025-12-31` | Seed: `42`")
    except Exception as exc:
        st.error(f"Dataset error: {exc}")


# -------------------------------------------------------------------------
# TOP COMMAND HEADER
# -------------------------------------------------------------------------
st.markdown(
    """
    <div class="ops-header">
        <div>
            <h1>VOLT<span>LINE</span> SMART SERVICE OPS CONSOLE</h1>
            <p>Real-Time NLP Triage, Dynamic Profit-Maximizing Quoting, Workforce Forecasting & Customer Retention</p>
        </div>
        <div class="telemetry-badge">
            CONTRACT v1.0 · CLASSICAL ML ENGINE · 3 BRANCHES (NCR · PUN · JAI)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_triage, tab_demand, tab_customers, tab_models = st.tabs(
    ["Triage & Quote", "Demand", "Customers", "Models & Impact"]
)


# =========================================================================
# TAB 1: TRIAGE & QUOTE
# =========================================================================
with tab_triage:
    if "sample_text" not in st.session_state:
        st.session_state.sample_text = SAMPLE_COMPLAINTS[0][1]
        st.session_state.sample_app = SAMPLE_COMPLAINTS[0][2]
        st.session_state.sample_prem = SAMPLE_COMPLAINTS[0][3]

    st.markdown("##### ⚡ Quick-Load Field Inquiry Scenarios (English + Hinglish)")
    chip_cols = st.columns(len(SAMPLE_COMPLAINTS))
    for idx, (label, s_txt, s_app, s_prem) in enumerate(SAMPLE_COMPLAINTS):
        if chip_cols[idx].button(label, key=f"sample_btn_{idx}", use_container_width=True):
            st.session_state.sample_text = s_txt
            st.session_state.sample_app = s_app
            st.session_state.sample_prem = s_prem

    col_in, col_out = st.columns([1.05, 2.15], gap="large")

    with col_in:
        st.markdown("#### 📞 Incoming Customer Inquiry")
        complaint_text = st.text_area("Customer Complaint", value=st.session_state.sample_text, height=95)
        c_a, c_b = st.columns(2)
        with c_a:
            appliance_type = st.selectbox("Appliance Type", options=APPLIANCES, index=0)
            branch_id = st.selectbox("Operating Branch", options=BRANCHES, index=0)
            source_channel = st.selectbox("Source Channel", options=CHANNELS, index=0)
        with c_b:
            premises_type = st.selectbox("Premises Type", options=PREMISES, index=0)
            skill_level = st.selectbox("Technician Skill Tier", options=[1, 2, 3], index=1)
            month = st.selectbox("Service Month", options=list(range(1, 13)), index=4)

        house_age_years = st.slider("Property Age (Years)", 0, 50, 18)
        experience_years = st.slider("Assigned Tech Experience (Years)", 1, 15, 5)
        response_time_min = st.slider("Lead Response Latency (Minutes)", 2, 180, 20, step=2)
        hour = st.number_input("Inquiry Hour (0-23)", min_value=0, max_value=23, value=14)

    with col_out:
        m1_result = None
        try:
            predict_triage_fn = load_m1_fn()
            m1_result = predict_triage_fn(
                text=complaint_text, appliance_type=appliance_type,
                premises_type=premises_type, house_age_years=int(house_age_years), hour=int(hour)
            )
        except Exception:
            st.warning("M1 Triage model not loaded.")

        if m1_result is not None:
            pred_fault = m1_result["fault"]
            pred_urg = m1_result["urgency"]
            fault_conf = m1_result["fault_probs"].get(pred_fault, 0.0)
            parts_pred = m1_result["parts"]

            m2_result = None
            m3b_result = None
            try:
                predict_job_fn = load_m2_fn()
                m2_result = predict_job_fn(
                    fault_category=pred_fault, appliance_type=appliance_type, premises_type=premises_type,
                    house_age_years=int(house_age_years), urgency=pred_urg, branch_id=branch_id,
                    experience_years=int(experience_years), skill_level=int(skill_level), month=int(month)
                )
                recommend_quote_fn = load_m3b_fn()
                m3b_result = recommend_quote_fn(
                    fault_category=pred_fault, appliance_type=appliance_type, premises_type=premises_type,
                    house_age_years=int(house_age_years), urgency=pred_urg, branch_id=branch_id,
                    experience_years=int(experience_years), skill_level=int(skill_level), month=int(month),
                    source_channel=source_channel, response_time_min=int(response_time_min), hour=int(hour), day_of_week=1
                )
            except Exception:
                pass

            dur_str = f"{m2_result['duration_hours']:.1f} h" if m2_result else "N/A"
            cost_str = f"₹{m2_result['cost_inr']:,.0f}" if m2_result else "N/A"
            quote_str = f"₹{m3b_result['price']:,.0f}" if m3b_result else "N/A"

            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Predicted Fault (M1)</div>
                        <div class="kpi-val">{FAULT_LABELS.get(pred_fault, pred_fault.upper())}</div>
                        <div class="kpi-sub mono-num">{fault_conf*100:.1f}% confidence</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Dispatch Urgency (M1)</div>
                        <div class="kpi-val"><span class="phase-pill u-{pred_urg}"><i class="phase-dot"></i>{pred_urg.upper()}</span></div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Est. Duration (M2)</div>
                        <div class="kpi-val mono-num">{dur_str}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Business Cost (M2)</div>
                        <div class="kpi-val mono-num">{cost_str}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Optimal Quote (M3B)</div>
                        <div class="kpi-val mono-num" style="color:#E3804B;">{quote_str}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            annotated_html, _, _, _ = highlight_xai_tokens(complaint_text)
            st.markdown("##### 🔍 Explainable NLP Signal Scanner")
            st.markdown(f'<div class="xai-box">{annotated_html}</div>', unsafe_allow_html=True)


# =========================================================================
# TAB 2: DEMAND
# =========================================================================
with tab_demand:
    st.markdown("#### 📈 Weekly Branch Demand Forecasting")
    sel_branch = st.selectbox("Select Branch", options=BRANCHES)
    try:
        forecast_fn = load_m4b_fn()
        fc_df = forecast_fn(branch_id=sel_branch, weeks=4)
        st.dataframe(fc_df, use_container_width=True)
    except Exception as e:
        st.info(f"Demand model running fallback / pending: {e}")


# =========================================================================
# TAB 3: CUSTOMERS
# =========================================================================
with tab_customers:
    st.markdown("#### 👥 RFM Customer Segmentation")
    try:
        seg_fn = load_m4a_fn()
        seg_df = seg_fn(as_of="2025-06-30")
        st.dataframe(seg_df.head(25), use_container_width=True)
    except Exception as e:
        st.info(f"Segmentation model pending: {e}")


# =========================================================================
# TAB 4: MODELS & IMPACT
# =========================================================================
with tab_models:
    st.markdown("#### 🏆 Unified Model Scoreboard & Diagnostic Figures")
    all_metrics = load_all_metrics()
    if all_metrics:
        st.json(all_metrics)
    
    fig_dir = ROOT_DIR / "reports" / "figures"
    png_files = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if png_files:
        for fig_path in png_files:
            st.image(str(fig_path), caption=fig_path.name, use_column_width=True)
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
.u-ok { color: #10B981; background: rgba(16, 185, 129, 0.12); }

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
# CACHED DATA & RESOURCE LOADERS (Contract Section 5 & 6)
# -------------------------------------------------------------------------
@st.cache_data
def get_dataset(name: str) -> pd.DataFrame:
    """Loads and caches a processed dataset via src.io.load."""
    return load(name)


@st.cache_data
def load_all_metrics() -> Dict[str, Dict[str, Any]]:
    """Reads all JSON files in reports/metrics/*.json."""
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
    """Checks which trained model artifacts exist in models/."""
    m_dir = ROOT_DIR / "models"
    return {
        "M1 (Triage)": (m_dir / "m1_fault.joblib").exists()
        and (m_dir / "m1_urgency.joblib").exists()
        and (m_dir / "m1_parts.joblib").exists(),
        "M2 (Estimator)": (m_dir / "m2_duration.joblib").exists()
        and (m_dir / "m2_cost.joblib").exists(),
        "M3A (Leads)": (m_dir / "m3a_lead.joblib").exists(),
        "M3B (Quote)": (ROOT_DIR / "src" / "m3b_quote.py").exists()
        and (m_dir / "m2_cost.joblib").exists()
        and (m_dir / "m3a_lead.joblib").exists(),
        "M4A (Segments)": (m_dir / "m4a_kmeans.joblib").exists()
        and (m_dir / "m4a_repeat.joblib").exists(),
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
    "copper_wire": 420,
    "connectors_tape": 90,
    "switch_socket": 180,
    "mcb": 380,
    "rccb": 1100,
    "cable_lugs": 60,
    "run_capacitor": 520,
    "contactor": 650,
    "isolator": 300,
    "thermostat": 350,
    "heating_element": 700,
    "geyser_plug": 150,
    "fan_capacitor": 60,
    "regulator": 220,
    "bearing_set": 180,
    "battery_terminal": 200,
    "fuse": 50,
    "led_driver": 250,
}

SAMPLE_COMPLAINTS = [
    (
        "⚡ Sparking & Smoke (Wiring)",
        "Main wiring me sparking ho rahi hai and burning smell aa rahi hai",
        "switchboard",
        "apartment",
    ),
    (
        "❄️ AC Tripping MCB",
        "Outdoor unit making heavy humming noise and MCB trips immediately",
        "ac",
        "house",
    ),
    (
        "🔥 Geyser Plug Melted",
        "Geyser switch board sparked and plug melted completely, garam pani nahi aa raha",
        "geyser",
        "apartment",
    ),
    (
        "🔋 Inverter No Backup",
        "Inverter continuous long beep de raha hai aur home light nahi hai during cut",
        "inverter_ups",
        "shop_office",
    ),
    (
        "🌀 Fan Slow & Humming",
        "Ceiling fan moving very slow even at highest regulator speed 5 and making noise",
        "fan_cooler",
        "apartment",
    ),
]

DANGER_PHRASES = [
    "burning smell",
    "jalne ki smell",
    "melting wire",
    "sparking",
    "spark",
    "smoke",
    "dhuan",
    "shock",
    "melted",
    "burnt",
]
POWER_LOSS_PHRASES = ["light nahi hai", "no power", "blackout", "completely dead"]
MILD_PHRASES = ["flickering", "slow", "noise", "dim", "humming"]


def highlight_xai_tokens(text: str) -> Tuple[str, List[str], List[str], List[str]]:
    """Highlights danger, power-loss, and mild diagnostic triggers in the complaint text."""
    found_danger, found_power, found_mild = [], [], []
    annotated = text

    for p in DANGER_PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE):
            found_danger.append(p)
            annotated = re.sub(
                rf"(\b{re.escape(p)}\b)",
                r'<span class="tok-danger">🚨 \1</span>',
                annotated,
                flags=re.IGNORECASE,
            )
    for p in POWER_LOSS_PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE):
            found_power.append(p)
            annotated = re.sub(
                rf"(\b{re.escape(p)}\b)",
                r'<span class="tok-power">⚡ \1</span>',
                annotated,
                flags=re.IGNORECASE,
            )
    for p in MILD_PHRASES:
        if re.search(rf"\b{re.escape(p)}\b", text, flags=re.IGNORECASE):
            found_mild.append(p)
            annotated = re.sub(
                rf"(\b{re.escape(p)}\b)",
                r'<span class="tok-mild">ℹ️ \1</span>',
                annotated,
                flags=re.IGNORECASE,
            )
    return annotated, found_danger, found_power, found_mild


def fallback_estimator_and_quote(
    fault_category: str,
    premises_type: str,
    house_age_years: int,
    urgency: str,
    experience_years: int,
    source_channel: str,
    response_time_min: int,
    parts_list: List[Dict[str, Any]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Contract-exact analytical physics engine used when M2/M3B .joblib files are not yet merged."""
    base_hours = {
        "wiring": 2.6,
        "mcb": 1.4,
        "ac": 2.2,
        "geyser": 1.6,
        "fan": 1.1,
        "inverter": 1.8,
        "light": 0.8,
    }
    prem_factor = {"apartment": 1.0, "house": 1.15, "shop_office": 1.25}
    chan_mult = {
        "google_ads": 0.9,
        "referral": 1.2,
        "website": 1.0,
        "whatsapp": 1.1,
        "walk_in": 1.3,
    }

    dur = (
        base_hours.get(fault_category, 1.6)
        * (1.0 + 0.012 * max(0, house_age_years - 10))
        * (1.2 - 0.035 * min(experience_years, 10))
        * prem_factor.get(premises_type, 1.0)
    )
    dur_low = round(dur * 0.82, 2)
    dur_high = round(dur * 1.22, 2)
    dur = round(dur, 2)

    parts_expected_cost = sum(
        p["prob"] * PART_PRICES_INR.get(p["part"], 200) for p in parts_list
    )
    cost_inr = round(380.0 * dur + parts_expected_cost + 199.0, 2)

    rc = get_dataset("rate_card")
    row = rc[
        (rc["fault_category"] == fault_category)
        & (rc["premises_type"] == premises_type)
    ]
    base_rate = float(row["base_price_inr"].iloc[0]) if not row.empty else cost_inr * 1.55
    typical_price = round(base_rate * (1.12 if urgency == "high" else 1.0), 2)

    resp_f = 0.6 + 0.4 * math.exp(-response_time_min / 70.0)
    k = 4.2 if urgency == "high" else 6.0
    c_mult = chan_mult.get(source_channel, 1.0)

    p_min = int(round((1.00 * cost_inr) / 5.0) * 5)
    p_max = int(round((2.60 * cost_inr) / 5.0) * 5)

    curve = []
    best_pt = None
    for price in range(p_min, p_max + 5, 5):
        q_ratio = price / typical_price
        win_p = min(
            1.0,
            max(0.0, (c_mult * resp_f) / (1.0 + math.exp(k * (q_ratio - 1.08)))),
        )
        exp_prof = win_p * (price - cost_inr)
        pt = {
            "price": float(price),
            "win_prob": round(float(win_p), 4),
            "expected_profit": round(float(exp_prof), 2),
        }
        curve.append(pt)
        if best_pt is None or pt["expected_profit"] > best_pt["expected_profit"]:
            best_pt = pt

    m2_out = {
        "duration_hours": dur,
        "duration_low": dur_low,
        "duration_high": dur_high,
        "cost_inr": cost_inr,
    }
    m3b_out = {
        "price": best_pt["price"],
        "win_prob": best_pt["win_prob"],
        "expected_profit": best_pt["expected_profit"],
        "cost_inr": cost_inr,
        "typical_price": typical_price,
        "curve": curve,
    }
    return m2_out, m3b_out


# -------------------------------------------------------------------------
# SIDEBAR — SYSTEM TELEMETRY & SIMULATION CONTROLS
# -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ VOLTLINE TELEMETRY")
    st.caption("Multi-Branch Franchise ML Control Plane")

    mod_status = get_module_status()
    online_count = sum(1 for v in mod_status.values() if v)
    st.progress(online_count / len(mod_status), text=f"Pipeline Modules Ready: {online_count}/{len(mod_status)}")

    for m_name, is_ready in mod_status.items():
        badge = "🟢 ONLINE" if is_ready else "🟡 PENDING (.joblib)"
        st.markdown(f"- **{m_name}**: `{badge}`")

    st.divider()
    st.markdown("### 🧪 PRESENTATION MODE")
    enable_sandbox = st.toggle(
        "Enable Physics Sandbox for Unmerged Modules",
        value=True,
        help="When enabled, any teammate module whose .joblib file is not yet trained will display the contract-required 'Module not available yet' notice AND render an interactive analytical preview using the Contract Section 5 equations.",
    )

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

# Exact 4 tab names required by Contract Section 5 (M5)
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
        complaint_text = st.text_area(
            "Customer Complaint (English / Hinglish)",
            value=st.session_state.sample_text,
            height=95,
        )

        c_a, c_b = st.columns(2)
        with c_a:
            appliance_type = st.selectbox(
                "Appliance Type",
                options=APPLIANCES,
                index=APPLIANCES.index(st.session_state.sample_app)
                if st.session_state.sample_app in APPLIANCES
                else 0,
            )
            branch_id = st.selectbox("Operating Branch", options=BRANCHES, index=0)
            source_channel = st.selectbox("Source Channel", options=CHANNELS, index=0)
        with c_b:
            premises_type = st.selectbox(
                "Premises Type",
                options=PREMISES,
                index=PREMISES.index(st.session_state.sample_prem)
                if st.session_state.sample_prem in PREMISES
                else 0,
            )
            skill_level = st.selectbox(
                "Technician Skill Tier",
                options=[1, 2, 3],
                index=1,
                format_func=lambda x: f"Tier {x} ({'Junior' if x==1 else 'Standard' if x==2 else 'Master'})",
            )
            month = st.selectbox(
                "Service Month",
                options=list(range(1, 13)),
                index=4,
                format_func=lambda m: pd.Timestamp(2025, m, 1).strftime("%B (%m)"),
            )

        house_age_years = st.slider("Property Age (Years)", 0, 50, 18)
        experience_years = st.slider("Assigned Tech Experience (Years)", 1, 15, 5)
        response_time_min = st.slider("Lead Response Latency (Minutes)", 2, 180, 20, step=2)

        c_h, c_d = st.columns(2)
        with c_h:
            hour = st.number_input("Inquiry Hour (0-23)", min_value=0, max_value=23, value=14)
        with c_d:
            day_of_week = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                index=1,
                format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
            )

    with col_out:
        # 1. Execute M1 Triage
        m1_result = None
        try:
            predict_triage_fn = load_m1_fn()
            m1_result = predict_triage_fn(
                text=complaint_text,
                appliance_type=appliance_type,
                premises_type=premises_type,
                house_age_years=int(house_age_years),
                hour=int(hour),
            )
        except Exception:
            st.warning("Module not available yet (M1: Job Triage)")

        if m1_result is not None:
            pred_fault = m1_result["fault"]
            pred_urg = m1_result["urgency"]
            fault_conf = m1_result["fault_probs"].get(pred_fault, 0.0)
            parts_pred = m1_result["parts"]

            # 2. Execute M2 & M3B if available, otherwise show contract notice
            m2_result = None
            m3b_result = None
            m2_m3b_live = True
            try:
                predict_job_fn = load_m2_fn()
                m2_result = predict_job_fn(
                    fault_category=pred_fault,
                    appliance_type=appliance_type,
                    premises_type=premises_type,
                    house_age_years=int(house_age_years),
                    urgency=pred_urg,
                    branch_id=branch_id,
                    experience_years=int(experience_years),
                    skill_level=int(skill_level),
                    month=int(month),
                )
                recommend_quote_fn = load_m3b_fn()
                m3b_result = recommend_quote_fn(
                    fault_category=pred_fault,
                    appliance_type=appliance_type,
                    premises_type=premises_type,
                    house_age_years=int(house_age_years),
                    urgency=pred_urg,
                    branch_id=branch_id,
                    experience_years=int(experience_years),
                    skill_level=int(skill_level),
                    month=int(month),
                    source_channel=source_channel,
                    response_time_min=int(response_time_min),
                    hour=int(hour),
                    day_of_week=int(day_of_week),
                )
            except Exception:
                m2_m3b_live = False
                st.info(
                    "ℹ️ **M2 / M3B Status:** `Module not available yet` — Waiting for teammate `.joblib` models (`m2_cost.joblib`, `m3a_lead.joblib`)."
                )
                if enable_sandbox:
                    m2_result, m3b_result = fallback_estimator_and_quote(
                        fault_category=pred_fault,
                        premises_type=premises_type,
                        house_age_years=int(house_age_years),
                        urgency=pred_urg,
                        experience_years=int(experience_years),
                        source_channel=source_channel,
                        response_time_min=int(response_time_min),
                        parts_list=parts_pred,
                    )

            # Top Telemetry Strip
            u_cls = {"high": "u-high", "medium": "u-medium", "low": "u-low"}.get(pred_urg, "u-low")
            sla_txt = {
                "high": "Immediate (<45m SLA)",
                "medium": "Same-Day (<4h SLA)",
                "low": "Next Standard Slot",
            }.get(pred_urg, "Standard")

            dur_str = f"{m2_result['duration_hours']:.1f} h" if m2_result else "N/A"
            dur_sub = (
                f"P10–P90: {m2_result['duration_low']:.1f}–{m2_result['duration_high']:.1f} h"
                if m2_result
                else "Module not available yet"
            )
            cost_str = f"₹{m2_result['cost_inr']:,.0f}" if m2_result else "N/A"
            quote_str = f"₹{m3b_result['price']:,.0f}" if m3b_result else "N/A"
            win_sub = (
                f"{m3b_result['win_prob']*100:.1f}% Win · ₹{m3b_result['expected_profit']:,.0f} Exp. Profit"
                if m3b_result
                else "Module not available yet"
            )

            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Predicted Fault (M1)</div>
                        <div class="kpi-val">{FAULT_LABELS.get(pred_fault, pred_fault.upper())}</div>
                        <div class="kpi-sub mono-num">{fault_conf*100:.1f}% model confidence</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Dispatch Urgency (M1)</div>
                        <div class="kpi-val"><span class="phase-pill {u_cls}"><i class="phase-dot"></i>{pred_urg.upper()}</span></div>
                        <div class="kpi-sub">{sla_txt}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Est. Duration (M2)</div>
                        <div class="kpi-val mono-num">{dur_str}</div>
                        <div class="kpi-sub mono-num">{dur_sub}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Business Cost (M2)</div>
                        <div class="kpi-val mono-num">{cost_str}</div>
                        <div class="kpi-sub">Labour + Parts + ₹199 visit</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-eyebrow">Optimal Quote (M3B)</div>
                        <div class="kpi-val mono-num" style="color:#E3804B;">{quote_str}</div>
                        <div class="kpi-sub mono-num">{win_sub}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explainable NLP Token Scanner
            annotated_html, d_toks, p_toks, m_toks = highlight_xai_tokens(complaint_text)
            st.markdown("##### 🔍 Explainable NLP Signal Scanner (Live Token Attribution)")
            st.markdown(
                f'<div class="xai-box">{annotated_html}</div>',
                unsafe_allow_html=True,
            )

            # Fault Probabilities & Van Loadout Side-by-Side
            c_f1, c_f2 = st.columns([1.15, 1.0], gap="medium")
            with c_f1:
                st.markdown("##### 📊 Multi-Class Fault Probability Distribution")
                probs_df = (
                    pd.DataFrame(
                        [
                            {
                                "Fault": FAULT_LABELS.get(k, k),
                                "Probability": v * 100.0,
                                "Code": k,
                            }
                            for k, v in m1_result["fault_probs"].items()
                        ]
                    )
                    .sort_values("Probability", ascending=True)
                )
                colors = [
                    "#E3804B" if code == pred_fault else "#475569"
                    for code in probs_df["Code"]
                ]
                fig_f = go.Figure(
                    go.Bar(
                        x=probs_df["Probability"],
                        y=probs_df["Fault"],
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in probs_df["Probability"]],
                        textposition="outside",
                    )
                )
                fig_f.update_layout(
                    height=250,
                    margin=dict(l=10, r=45, t=10, b=10),
                    xaxis=dict(title="Confidence (%)", range=[0, 112], showgrid=True),
                    yaxis=dict(title=""),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_f, use_container_width=True)

            with c_f2:
                st.markdown("##### 🚐 Van Spare-Parts Loadout Manifest (Top 6)")
                chips_html = '<div class="parts-wrap">'
                carry_count = 0
                est_inv_value = 0
                for item in parts_pred:
                    p_code = item["part"]
                    p_prob = item["prob"]
                    is_carry = p_prob >= 0.35
                    if is_carry:
                        carry_count += 1
                        est_inv_value += PART_PRICES_INR.get(p_code, 200)
                    cls_name = "part-chip carry" if is_carry else "part-chip"
                    prefix = "✅ LOAD: " if is_carry else ""
                    chips_html += (
                        f'<div class="{cls_name}">{prefix}{p_code.replace("_", " ").title()}'
                        f'<span>{p_prob*100:.0f}% (₹{PART_PRICES_INR.get(p_code, 0)})</span></div>'
                    )
                chips_html += "</div>"
                st.markdown(chips_html, unsafe_allow_html=True)
                st.caption(
                    f"Outlined parts (`≥35%` probability) are flagged for van loading to prevent a repeat visit. "
                    f"**Recommended Van Kit:** `{carry_count} items` (Kit Value: `₹{est_inv_value:,}`)."
                )

                # Urgency probability breakdown mini-bar
                urg_df = pd.DataFrame(
                    [
                        {"Level": k.upper(), "Prob": v * 100.0}
                        for k, v in m1_result["urgency_probs"].items()
                    ]
                )
                st.dataframe(
                    urg_df.style.format({"Prob": "{:.1f}%"}),
                    use_container_width=True,
                    hide_index=True,
                )

            # 3. Quote Optimizer Interactive Curve
            st.markdown("---")
            hdr_q1, hdr_q2 = st.columns([2, 1])
            with hdr_q1:
                st.markdown("##### 💰 Dynamic Quote Optimizer — Profit vs. Conversion Elasticity Curve")
            with hdr_q2:
                if not m2_m3b_live and enable_sandbox:
                    st.caption("🧪 *Running Contract Analytical Sandbox*")

            if m3b_result is not None:
                curve_df = pd.DataFrame(m3b_result["curve"])
                opt_price = m3b_result["price"]
                opt_profit = m3b_result["expected_profit"]
                opt_win = m3b_result["win_prob"]
                cost_val = m3b_result["cost_inr"]
                typ_val = m3b_result["typical_price"]
                baseline_price = round((cost_val * 1.30) / 5.0) * 5.0

                # Interactive price explorer slider
                selected_quote = st.slider(
                    "Explore Candidate Quote Price (INR)",
                    min_value=int(curve_df["price"].min()),
                    max_value=int(curve_df["price"].max()),
                    value=int(opt_price),
                    step=5,
                )

                sel_row = curve_df.iloc[(curve_df["price"] - selected_quote).abs().argsort()[:1]].iloc[0]
                base_row = curve_df.iloc[(curve_df["price"] - baseline_price).abs().argsort()[:1]].iloc[0]
                lift_pct = (
                    ((opt_profit - base_row["expected_profit"]) / base_row["expected_profit"]) * 100.0
                    if base_row["expected_profit"] > 0
                    else 0.0
                )

                fig_q = make_subplots(specs=[[{"secondary_y": True}]])
                fig_q.add_trace(
                    go.Scatter(
                        x=curve_df["price"],
                        y=curve_df["expected_profit"],
                        name="Expected Profit (INR)",
                        mode="lines",
                        line=dict(color="#E3804B", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(227, 128, 75, 0.14)",
                    ),
                    secondary_y=False,
                )
                fig_q.add_trace(
                    go.Scatter(
                        x=curve_df["price"],
                        y=curve_df["win_prob"] * 100.0,
                        name="Win Probability (%)",
                        mode="lines",
                        line=dict(color="#3B82F6", width=2.2, dash="dash"),
                    ),
                    secondary_y=True,
                )
                # Optimal point marker
                fig_q.add_trace(
                    go.Scatter(
                        x=[opt_price],
                        y=[opt_profit],
                        name=f"Optimal Quote (₹{opt_price:.0f})",
                        mode="markers+text",
                        marker=dict(color="#10B981", size=12, line=dict(color="#FFFFFF", width=2)),
                        text=[f"Optimal: ₹{opt_price:.0f}"],
                        textposition="top center",
                    ),
                    secondary_y=False,
                )
                # Baseline Cost + 30% vertical line
                fig_q.add_vline(
                    x=baseline_price,
                    line_dash="dot",
                    line_color="#94A3B8",
                    annotation_text=f"Cost+30% (₹{baseline_price:.0f})",
                    annotation_position="bottom right",
                )
                # User selected quote line
                fig_q.add_vline(
                    x=selected_quote,
                    line_dash="solid",
                    line_color="#F59E0B",
                    annotation_text=f"Selected (₹{selected_quote:.0f})",
                    annotation_position="top left",
                )

                fig_q.update_layout(
                    height=330,
                    margin=dict(l=15, r=15, t=20, b=15),
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                fig_q.update_xaxes(title_text="Candidate Quoted Price (INR)")
                fig_q.update_yaxes(title_text="Expected Profit per Lead (INR)", secondary_y=False)
                fig_q.update_yaxes(title_text="Booking Probability (%)", range=[0, 105], secondary_y=True)
                st.plotly_chart(fig_q, use_container_width=True)

                st.markdown(
                    f"**Quote Telemetry Readout:** At **₹{selected_quote:,.0f}** "
                    f"(`quote_ratio = {selected_quote/typ_val:.2f}` against typical price `₹{typ_val:,.0f}`), "
                    f"conversion probability is **{sel_row['win_prob']*100:.1f}%** with an expected profit of "
                    f"**₹{sel_row['expected_profit']:,.0f}/lead**. "
                    f"Compared to the flat `Cost + 30%` rule (`₹{baseline_price:,.0f}` → `₹{base_row['expected_profit']:,.0f}`), "
                    f"the M3B optimal price (`₹{opt_price:,.0f}`) delivers a **{lift_pct:+.1f}% profit lift**."
                )


# =========================================================================
# TAB 2: DEMAND
# =========================================================================
with tab_demand:
    st.markdown("#### 📈 Weekly Branch Demand Forecasting & Workforce Capacity Planner")
    c_d1, c_d2 = st.columns([1, 3])
    with c_d1:
        sel_branch = st.selectbox(
            "Select Franchise Branch",
            options=BRANCHES,
            format_func=lambda b: {"NCR": "NCR — Delhi NCR (9 Techs)", "PUN": "PUN — Pune (6 Techs)", "JAI": "JAI — Jaipur (5 Techs)"}[b],
        )
        horizon_weeks = st.slider("Forecast Horizon (Weeks)", min_value=1, max_value=8, value=4)

    branches_df = get_dataset("branches")
    jobs_df = get_dataset("jobs")
    weather_df = get_dataset("weather")

    on_staff = int(
        branches_df.loc[branches_df["branch_id"] == sel_branch, "n_technicians"].iloc[0]
    )

    # Historical Weekly Actuals per Branch (Complete Monday-Sunday weeks)
    b_jobs = jobs_df[jobs_df["branch_id"] == sel_branch].copy()
    b_jobs["week_start"] = b_jobs["job_date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
    weekly_actuals = (
        b_jobs.groupby("week_start")
        .size()
        .reset_index(name="actual_jobs")
        .sort_values("week_start")
    )
    weekly_actuals = weekly_actuals[
        (weekly_actuals["week_start"] >= pd.Timestamp("2024-01-01"))
        & (weekly_actuals["week_start"] <= pd.Timestamp("2025-12-22"))
    ]

    # Weather weekly aggregation for visual overlay
    b_weather = weather_df[weather_df["branch_id"] == sel_branch].copy()
    b_weather["week_start"] = b_weather["date"].dt.to_period("W-SUN").apply(lambda r: r.start_time)
    weekly_weather = (
        b_weather.groupby("week_start")
        .agg(temp_mean=("temp_c", "mean"), rain_sum=("rain_mm", "sum"), fest=("festival_flag", "max"))
        .reset_index()
    )
    weekly_actuals = weekly_actuals.merge(weekly_weather, on="week_start", how="left")

    # Call M4B forecast_demand
    fc_df = None
    m4b_live = True
    try:
        forecast_demand_fn = load_m4b_fn()
        fc_df = forecast_demand_fn(branch_id=sel_branch, weeks=int(horizon_weeks))
    except Exception:
        m4b_live = False
        st.info("ℹ️ **M4B Status:** `Module not available yet` — Waiting for `models/m4b_demand.joblib`.")
        if enable_sandbox:
            last_4_mean = weekly_actuals["actual_jobs"].tail(4).mean()
            season_dec_jan = [0.90, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
            rows_fc = []
            for h in range(1, int(horizon_weeks) + 1):
                ws = pd.Timestamp("2025-12-29") + pd.Timedelta(days=7 * (h - 1))
                f_val = round(float(last_4_mean * season_dec_jan[h - 1] / 0.90), 1)
                band = 0.08 + 0.02 * h
                rows_fc.append(
                    {
                        "week_start": ws,
                        "forecast": f_val,
                        "lower": round(f_val * (1.0 - band), 1),
                        "upper": round(f_val * (1.0 + band), 1),
                        "techs_needed": int(math.ceil(f_val / JOBS_PER_TECH_PER_WEEK)),
                    }
                )
            fc_df = pd.DataFrame(rows_fc)

    if fc_df is not None:
        fc_df["week_start"] = pd.to_datetime(fc_df["week_start"])
        recent_actuals = weekly_actuals.tail(32).copy()

        fig_d = make_subplots(specs=[[{"secondary_y": True}]])
        # Confidence Interval Band
        fig_d.add_trace(
            go.Scatter(
                x=pd.concat([fc_df["week_start"], fc_df["week_start"][::-1]]),
                y=pd.concat([fc_df["upper"], fc_df["lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(227, 128, 75, 0.22)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Forecast Interval (P10–P90)",
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        # Actuals Line
        fig_d.add_trace(
            go.Scatter(
                x=recent_actuals["week_start"],
                y=recent_actuals["actual_jobs"],
                name="Actual Weekly Jobs",
                mode="lines+markers",
                line=dict(color="#E2E8F0", width=2.5),
                marker=dict(size=5),
            ),
            secondary_y=False,
        )
        # Forecast Line
        connect_x = [recent_actuals["week_start"].iloc[-1]] + list(fc_df["week_start"])
        connect_y = [recent_actuals["actual_jobs"].iloc[-1]] + list(fc_df["forecast"])
        fig_d.add_trace(
            go.Scatter(
                x=connect_x,
                y=connect_y,
                name="4-Week Recursive Forecast",
                mode="lines+markers",
                line=dict(color="#E3804B", width=3, dash="dash"),
                marker=dict(size=8, color="#E3804B"),
            ),
            secondary_y=False,
        )
        # Capacity Line (On-Staff * 13 jobs/week)
        staff_cap = on_staff * JOBS_PER_TECH_PER_WEEK
        fig_d.add_hline(
            y=staff_cap,
            line_dash="dot",
            line_color="#10B981",
            annotation_text=f"Current Branch Capacity ({on_staff} Techs = {staff_cap} jobs/wk)",
            annotation_position="top left",
        )
        # Ambient Temperature overlay
        fig_d.add_trace(
            go.Scatter(
                x=recent_actuals["week_start"],
                y=recent_actuals["temp_mean"],
                name="Mean Temp (°C)",
                mode="lines",
                line=dict(color="#3B82F6", width=1.3, dash="dot"),
                opacity=0.65,
            ),
            secondary_y=True,
        )

        fig_d.update_layout(
            height=390,
            margin=dict(l=15, r=15, t=25, b=15),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_d.update_yaxes(title_text="Weekly Jobs Count", secondary_y=False)
        fig_d.update_yaxes(title_text="Ambient Temp (°C)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig_d, use_container_width=True)

        # Staffing Action Table
        st.markdown(f"##### 👷 Workforce & Shift Allocation Plan (`{JOBS_PER_TECH_PER_WEEK}` Jobs / Technician / Week)")
        staff_disp = fc_df.copy()
        staff_disp["week_start"] = staff_disp["week_start"].dt.strftime("%Y-%m-%d")
        staff_disp["on_staff"] = on_staff
        staff_disp["gap"] = staff_disp["techs_needed"] - on_staff
        staff_disp["utilization_pct"] = (
            staff_disp["forecast"] / (on_staff * JOBS_PER_TECH_PER_WEEK) * 100.0
        ).round(1)
        staff_disp["status"] = staff_disp["gap"].apply(
            lambda g: f"🔴 SHORT BY {g} TECH(S)"
            if g > 0
            else (f"🟢 OPTIMAL (SURPLUS {-g})" if g < 0 else "🟢 EXACT CAPACITY")
        )
        st.dataframe(
            staff_disp[
                [
                    "week_start",
                    "forecast",
                    "lower",
                    "upper",
                    "techs_needed",
                    "on_staff",
                    "utilization_pct",
                    "status",
                ]
            ].rename(
                columns={
                    "week_start": "Week Starting (Mon)",
                    "forecast": "Forecast Jobs",
                    "lower": "Lower Bound",
                    "upper": "Upper Bound",
                    "techs_needed": "Techs Needed",
                    "on_staff": "Techs On Staff",
                    "utilization_pct": "Staff Utilization (%)",
                    "status": "Dispatch Action Plan",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================================
# TAB 3: CUSTOMERS
# =========================================================================
with tab_customers:
    st.markdown("#### 👥 RFM Customer Segmentation & 6-Month Repeat Propensity Engine")
    as_of_date = st.selectbox(
        "RFM Snapshot Cutoff Date (`as_of`)",
        options=["2025-06-30", "2024-12-31", "2025-09-30", "2025-12-31"],
        index=0,
    )

    seg_df = None
    m4a_live = True
    try:
        get_segments_fn = load_m4a_fn()
        seg_df = get_segments_fn(as_of=as_of_date)
    except Exception:
        m4a_live = False
        st.info("ℹ️ **M4A Status:** `Module not available yet` — Waiting for `models/m4a_kmeans.joblib` and `models/m4a_repeat.joblib`.")
        if enable_sandbox:
            jobs_all = get_dataset("jobs")
            cutoff = pd.to_datetime(as_of_date)
            hist = jobs_all[jobs_all["job_date"] <= cutoff].copy()
            rfm_df = (
                hist.groupby("customer_id")
                .agg(
                    last_job=("job_date", "max"),
                    frequency=("job_id", "count"),
                    monetary=("final_price_inr", "sum"),
                )
                .reset_index()
            )
            rfm_df["recency_days"] = (cutoff - rfm_df["last_job"]).dt.days
            # Rule-based K-Means proxy following Contract Section 5
            conds = [
                (rfm_df["frequency"] >= 3) & (rfm_df["monetary"] >= rfm_df["monetary"].quantile(0.70)),
                (rfm_df["recency_days"] > 180) & (rfm_df["frequency"] >= 2),
                (rfm_df["frequency"] == 1),
            ]
            choices = ["loyal_high_value", "at_risk", "new_one_time"]
            rfm_df["segment"] = np.select(conds, choices, default="regular_seasonal")
            rfm_df["repeat_prob_6m"] = np.clip(
                0.22
                + 0.11 * rfm_df["frequency"]
                - 0.0006 * rfm_df["recency_days"]
                + np.where(rfm_df["segment"] == "loyal_high_value", 0.18, 0.0),
                0.03,
                0.96,
            ).round(4)
            seg_df = rfm_df[
                [
                    "customer_id",
                    "recency_days",
                    "frequency",
                    "monetary",
                    "segment",
                    "repeat_prob_6m",
                ]
            ]

    if seg_df is not None:
        ACTION_MAP = {
            "loyal_high_value": "Offer Annual Maintenance Contract (AMC) & Priority Dispatch",
            "regular_seasonal": "Send Pre-Summer AC / Pre-Winter Geyser Preventive Check",
            "new_one_time": "Request Google Review + ₹150 Safety Inspection Coupon",
            "at_risk": "Trigger Win-Back WhatsApp Discount (15% Off Next Visit)",
        }

        summary_tbl = (
            seg_df.groupby("segment")
            .agg(
                Customers=("customer_id", "count"),
                Avg_Recency_Days=("recency_days", "mean"),
                Avg_Frequency=("frequency", "mean"),
                Avg_Monetary_INR=("monetary", "mean"),
                Avg_Repeat_Prob=("repeat_prob_6m", "mean"),
            )
            .reset_index()
        )
        summary_tbl["Recommended CRM Action"] = summary_tbl["segment"].map(ACTION_MAP)

        st.markdown("##### 📋 Segment Cohort Summary")
        st.dataframe(
            summary_tbl.style.format(
                {
                    "Avg_Recency_Days": "{:.1f} d",
                    "Avg_Frequency": "{:.2f}",
                    "Avg_Monetary_INR": "₹{:,.0f}",
                    "Avg_Repeat_Prob": "{:.1%}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        c_s1, c_s2 = st.columns([1.35, 1.0], gap="large")
        with c_s1:
            st.markdown("##### 🎯 Interactive RFM Segment Explorer")
            sel_segs = st.multiselect(
                "Filter Segments",
                options=sorted(seg_df["segment"].unique()),
                default=sorted(seg_df["segment"].unique()),
            )
            plot_sample = seg_df[seg_df["segment"].isin(sel_segs)].sample(
                n=min(1200, len(seg_df[seg_df["segment"].isin(sel_segs)])),
                random_state=42,
            )
            fig_rfm = px.scatter(
                plot_sample,
                x="recency_days",
                y="frequency",
                size="monetary",
                color="segment",
                hover_data=["customer_id", "monetary", "repeat_prob_6m"],
                color_discrete_map={
                    "loyal_high_value": "#10B981",
                    "regular_seasonal": "#3B82F6",
                    "new_one_time": "#F59E0B",
                    "at_risk": "#EF4444",
                },
                labels={
                    "recency_days": "Recency (Days Since Last Job)",
                    "frequency": "Job Frequency (Count)",
                    "monetary": "Total Spend (INR)",
                },
            )
            fig_rfm.update_layout(
                height=370,
                margin=dict(l=10, r=10, t=15, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_rfm, use_container_width=True)

        with c_s2:
            st.markdown("##### 🔔 Proactive Maintenance Reminder Queue (Top 20)")
            st.caption("Contract Filter: `repeat_prob_6m > 0.5` and `recency_days > 90`")
            rem_df = (
                seg_df[
                    (seg_df["repeat_prob_6m"] > 0.5) & (seg_df["recency_days"] > 90)
                ]
                .sort_values("repeat_prob_6m", ascending=False)
                .head(20)
                .copy()
            )
            st.dataframe(
                rem_df.style.format(
                    {
                        "monetary": "₹{:,.0f}",
                        "repeat_prob_6m": "{:.1%}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
                height=295,
            )
            csv_bytes = rem_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Top-20 Maintenance Reminder List (CSV)",
                data=csv_bytes,
                file_name=f"voltline_reminders_{as_of_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )


# =========================================================================
# TAB 4: MODELS & IMPACT
# =========================================================================
with tab_models:
    st.markdown("#### 🏆 Unified Model Scoreboard & Business Impact Simulation")
    all_metrics = load_all_metrics()

    if not all_metrics:
        st.warning("No metrics JSON files found in `reports/metrics/`. Run training scripts or `python -m src.evaluate_impact`.")
    else:
        rows_score = []
        for mod_key, payload in sorted(all_metrics.items()):
            for m_entry in payload.get("models", []):
                prim = m_entry.get("primary_metric", {})
                base = m_entry.get("baseline", {})
                base_metrics = base.get("metrics", {})
                base_str = ", ".join(f"{k}: {v}" for k, v in base_metrics.items())
                sec_str = ", ".join(
                    f"{k}: {v}" for k, v in m_entry.get("metrics", {}).items()
                )
                rows_score.append(
                    {
                        "Module": payload.get("module", mod_key),
                        "Model Name": m_entry.get("name", ""),
                        "Task": m_entry.get("task", ""),
                        "Primary Metric": prim.get("name", ""),
                        "Test Score": prim.get("value", 0.0),
                        "Secondary Metrics": sec_str,
                        "Baseline Model": f"{base.get('name', '')} ({base_str})",
                        "Train / Val / Test": f"{m_entry.get('n_train', 0):,} / {m_entry.get('n_val', 0):,} / {m_entry.get('n_test', 0):,}",
                    }
                )
        if rows_score:
            st.dataframe(
                pd.DataFrame(rows_score),
                use_container_width=True,
                hide_index=True,
            )

    # Display Business Impact Comparison Cards (from M5 metrics JSON if available)
    if "M5" in all_metrics and "impact_summary" in all_metrics["M5"]:
        imp = all_metrics["M5"]["impact_summary"]
        st.markdown("##### 🚀 Simulated Business Impact on Held-Out Test Split (vs. No-ML Baseline)")
        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-eyebrow">1. Repeat-Visit Rate (M1 Parts)</div>
                    <div class="kpi-val mono-num">{imp['repeat_rate_baseline']*100:.2f}% → <span style="color:#10B981;">{imp['repeat_rate_ml']*100:.2f}%</span></div>
                    <div class="kpi-sub">Van top-6 parts coverage: {imp['parts_coverage_rate']*100:.1f}%</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-eyebrow">2. Expected Profit / Lead (M3B)</div>
                    <div class="kpi-val mono-num">₹{imp['profit_per_lead_baseline']:,.0f} → <span style="color:#10B981;">₹{imp['profit_per_lead_ml']:,.0f}</span></div>
                    <div class="kpi-sub">+{imp['profit_lift_pct']:.1f}% over flat Cost × 1.30</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-eyebrow">3. Top-30% Lead Conversions (M3A)</div>
                    <div class="kpi-val mono-num">{imp['top30_conv_random']:,} → <span style="color:#10B981;">{imp['top30_conv_ml']:,}</span></div>
                    <div class="kpi-sub">+{imp['top30_conv_lift_pct']:.1f}% prioritized bookings</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-eyebrow">4. Tech-Weeks Short/Idle (M4B)</div>
                    <div class="kpi-val mono-num">{imp['tech_weeks_mismatch_baseline']} → <span style="color:#10B981;">{imp['tech_weeks_mismatch_ml']}</span></div>
                    <div class="kpi-sub">vs. constant 2025 H1 mean staffing</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("📖 View Simulation Assumptions & Methodology (`reports/metrics/m5.json`)"):
            for idx_a, a_text in enumerate(all_metrics["M5"].get("assumptions", []), 1):
                st.markdown(f"{idx_a}. {a_text}")

    # Diagnostic Figures Gallery from reports/figures/*.png
    st.markdown("##### 🖼️ Diagnostic & Evaluation Figures Gallery (`reports/figures/`)")
    fig_dir = ROOT_DIR / "reports" / "figures"
    png_files = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
    if png_files:
        g_cols = st.columns(2)
        for idx_f, fig_path in enumerate(png_files):
            with g_cols[idx_f % 2]:
                st.image(str(fig_path), caption=fig_path.name, use_container_width=True)
    else:
        st.caption("No PNG figures found yet.")
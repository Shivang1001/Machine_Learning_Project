"""M0A: High-performance synthetic data generator implementing Voltline Contract v1.0 specifications."""

import datetime
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    RANDOM_STATE, DATA_START, DATA_END, FAULTS, URGENCY,
    APPLIANCES, PREMISES, CHANNELS, BRANCHES, PARTS
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(RANDOM_STATE)

PART_UNIT_PRICES = {
    "copper_wire": 420, "connectors_tape": 90, "switch_socket": 180, "mcb": 380,
    "rccb": 1100, "cable_lugs": 60, "run_capacitor": 520, "contactor": 650,
    "isolator": 300, "thermostat": 350, "heating_element": 700, "geyser_plug": 150,
    "fan_capacitor": 60, "regulator": 220, "bearing_set": 180, "battery_terminal": 200,
    "fuse": 50, "led_driver": 250
}

FAULT_PART_PROBS = {
    "wiring": [("copper_wire", 0.80), ("connectors_tape", 0.70), ("switch_socket", 0.45), ("mcb", 0.25)],
    "mcb": [("mcb", 0.70), ("rccb", 0.30), ("cable_lugs", 0.50)],
    "ac": [("run_capacitor", 0.55), ("contactor", 0.30), ("isolator", 0.35)],
    "geyser": [("thermostat", 0.60), ("heating_element", 0.45), ("geyser_plug", 0.30)],
    "fan": [("fan_capacitor", 0.60), ("regulator", 0.40), ("bearing_set", 0.30)],
    "inverter": [("battery_terminal", 0.40), ("fuse", 0.50)],
    "light": [("switch_socket", 0.60), ("connectors_tape", 0.40), ("led_driver", 0.30)]
}

COMPLAINT_TEMPLATES = {
    "wiring": [
        "Main wiring me sparking ho rahi hai and burning smell aa rahi hai",
        "There is continuous sparking from the distribution line and a smoke smell",
        "Short circuit happened in bedroom wiring, light nahi hai abhi",
        "Wire melting ho rahi hai near main switch, immediate help needed",
        "Complete wiring looks burnt and jalne ki smell is persistent",
        "Sudden electric shock felt from wall, wires might be damaged",
        "Short circuit in corridor, sparks flew out and tripped everything",
        "Heavy burning smell coming from behind the cabinet wiring",
        "Melting wire noticed inside bedroom conduit, high risk",
        "Power went off with a loud pop, visible smoke from wall conduit",
        "Frequent short circuit tripping, wires heat up very fast",
        "Severe current leakage on switch plate, getting minor shock",
        "Wiring inside meter box sparked with loud crackle",
        "Old concealed wiring giving burning smell across the hall",
        "Multiple wires melted together inside junction box, no power",
        "Dhuan nikal raha hai wiring conduit se, please send technician",
        "Sparks coming out continuously whenever AC is switched on",
        "Light chali gayi short circuit ke baad, smelling burnt plastic",
        "Main line wire burnt completely, emergency help required",
        "Electric current passing through bathroom tap and switch"
    ],
    "mcb": [
        "Main MCB tripping repeatedly every 10 minutes without heavy load",
        "Distribution board switch tripped with loud sound and won't reset",
        "MCB box se halki smoke smell aa rahi hai aur fuse switch loose hai",
        "RCCB tripping instantly whenever inverter or fridge turns on",
        "DB switchboard warm ho raha hai aur tripping issue ho raha hai",
        "Main circuit breaker won't stay in UP position, total power loss",
        "Frequent tripping of 32A breaker switchboard, light nahi hai",
        "MCB spark after power cut and now main power is cut off",
        "Distribution board making constant buzzing sound and tripping",
        "Sub-meter MCB tripped, burning plastic smell nearby",
        "One phase completely dead because of faulty distribution board switch",
        "RCCB drops immediately on turning on any appliance",
        "Breaker trips even when all room switches are kept off",
        "MCB switch jammed and sparks when forced up",
        "Heavy humming and switchboard crackling inside main DB box",
        "Light nahi hai because central MCB trip ho gaya aur uth nahi raha",
        "DB board neutral link burnt, tripping power to kitchen",
        "Circuit breaker heating up rapidly under normal load",
        "Tripping problem on main 63A switchboard, entire house dark",
        "Overload switch tripping without any heavy machine running"
    ],
    "ac": [
        "AC cooling band ho gayi hai completely, fan chal raha hai bas",
        "Split AC turning on but compressor not starting at all",
        "AC switchboard se smoke and burning smell came suddenly",
        "Outdoor unit making heavy humming noise and MCB trips immediately",
        "AC run capacitor burst sound heard, cooling totally stopped",
        "Window AC compressor tripping within 2 minutes of turning on",
        "AC outdoor unit not getting power, isolator switch issue",
        "Thanda nahi kar raha AC, air is normal room temp",
        "AC plug point burnt and melted wire visible inside 16A socket",
        "Strange clicking noise from AC PCB and no cooling output",
        "AC switch sparking whenever compressor tries to kick in",
        "AC stabilizing unit smells burnt and display is off",
        "Outdoor unit cooling coil fan running slow, tripping breaker",
        "Cooling failure in bedroom split unit, high temperature inside",
        "AC contactor vibrating loudly and outdoor unit refuses to run",
        "AC switchboard heated up and stopped supplying power to AC",
        "Compressor cut off after loud spark inside AC switch box",
        "Water leaking near AC electrical box, tripping circuit",
        "AC turns off automatically after 5 minutes with error beep",
        "AC is blowing room temperature air, compressor not firing"
    ],
    "geyser": [
        "Geyser se garam pani bilkul nahi aa raha, power lamp is on",
        "Water heater heating element seems dead, water is ice cold",
        "Geyser switch board sparked and plug melted completely",
        "Thermostat trip ho gaya hai lagta hai, water not heating",
        "Current coming in bathroom running water when geyser is switched on",
        "Geyser power light off, no hot water coming since morning",
        "Water heater element making boiling sound and dripping water",
        "Geyser 16A plug loose and heating up with jalne ki smell",
        "Water heater trips MCB the moment toggle is pressed on",
        "Thermostat not cutting off, geyser boiling violently and steaming",
        "Garam pani nahi aa raha geyser me se, winter emergency",
        "Electric shock felt while touching geyser metal pipe",
        "Geyser indicator is dead and reset button not helping",
        "Water heater power socket black and burnt around pins",
        "Geyser takes over 2 hours to provide mild lukewarm water",
        "Continuous humming and clicking from water heater circuit",
        "Geyser power cord charred near base entry point",
        "Complete power loss to water heater point in washroom",
        "Thermostat replacement needed for 25L storage water heater",
        "Geyser switch sparks whenever turned on, water not warm"
    ],
    "fan": [
        "Ceiling fan moving very slow even at highest regulator speed 5",
        "Fan making heavy grinding and humming noise from motor bearing",
        "Ceiling fan capacitor dead, needs push by hand to start rotating",
        "Fan regulator burning smell coming and knob is completely stuck",
        "Cooler motor humming loudly but blade not rotating at all",
        "Exhaust fan stopped spinning with minor smoke smell in kitchen",
        "Ceiling fan wobbling violently and regulator sparks on speed change",
        "Fan running dead slow, capacitor needs immediate replacement",
        "Room cooler water pump working but main motor is dead",
        "Fan making continuous squeaking noise, bearing set dry or damaged",
        "Fan stopped abruptly after a faint burning smell in bedroom",
        "Regulator burnt and fan stuck on maximum speed constantly",
        "Cooler motor sparking inside switch housing, urgent fix",
        "Ceiling fan speed drops to zero after 15 minutes of running",
        "Humming sound from fan motor but blades don't spin freely",
        "Fan capacitor leaking oil over motor body, running slow",
        "Fan regulator switchboard loose and sparking slightly",
        "Cooler blower fan jammed, motor smells hot and buzzes",
        "High ceiling fan speed fluctuation and regulator heating up",
        "Fan noise is loud and irritating at nights, speed very slow"
    ],
    "inverter": [
        "Inverter continuous long beep de raha hai aur backup nahi hai",
        "Inverter battery charging nahi le rahi, battery terminals corroded",
        "Power cut hote hi inverter trips, no power backup to home",
        "Inverter unit giving burning plastic smell and battery boiling warm",
        "Backup duration dropped from 4 hours to just 10 minutes",
        "Inverter display shows overload error even when no load connected",
        "Inverter main fuse blown, unit totally dead with no LED lights",
        "Heavy sulfuric battery smell and acid leakage around terminal",
        "Inverter not switching to mains power when electricity returns",
        "UPS beeping continuously, home light nahi hai during cut",
        "Inverter output voltage fluctuating heavily, lights flickering",
        "Battery terminal sparked while inverter was charging",
        "UPS trips the main DB switchboard immediately upon switching on",
        "Inverter humming loudly and cooling fan inside is not spinning",
        "No power switchover happening, inverter shuts off instantly",
        "Inverter blown internal fuse during thunder cut",
        "Battery water level normal but battery drains in 5 minutes",
        "Charging current not stabilizing, inverter body very hot",
        "Inverter circuit sparked and smells burnt, emergency backup needed",
        "Complete blackout in house during power outage, inverter dead"
    ],
    "light": [
        "Hall lights flickering continuously and switch board loose",
        "Living room switch sparked when turning on chandelier",
        "LED panel driver burnt, multiple false ceiling lights down",
        "Switch loose and making cracking noise on toggle, light dim",
        "Light bulb holder sparked and melted, light bulb fell down",
        "Dim lighting across bedroom, switch socket sparking lightly",
        "Wall switch board loose and shocks slightly on touch",
        "Ceiling LED strip driver stopped working, no lights in hall",
        "Multiple tube lights flickering together and buzzing softly",
        "Bathroom light switch stuck and sparks when pressed",
        "Kitchen light switch heated up and turned black around edges",
        "False ceiling spot light driver damaged, flickering badly",
        "Switchboard toggle cracked, socket not holding plug securely",
        "Exterior balcony light fixture dead, wires seem loose",
        "Two light points in corridor blinking randomly, very dim",
        "LED driver blast sound heard, false ceiling light dead",
        "Switch board plate vibrating slightly when tube light turns on",
        "Light holder damaged, bulb not making proper contact",
        "Frequent bulb fusing in bedroom, voltage seems irregular",
        "Wall bracket light switch stuck, light flickering on and off"
    ]
}

DANGER_WORDS = ["spark", "sparking", "smoke", "burning smell", "jalne ki smell", "shock", "melting wire"]
POWER_LOSS_WORDS = ["light nahi hai", "no power"]
MILD_WORDS = ["flickering", "slow", "noise"]

TEMPLATE_METADATA = {}
for f, tmpls in COMPLAINT_TEMPLATES.items():
    TEMPLATE_METADATA[f] = []
    for tmpl in tmpls:
        low = tmpl.lower()
        d_cnt = sum(1 for w in DANGER_WORDS if w in low)
        p_loss = 1 if any(w in low for w in POWER_LOSS_WORDS) else 0
        m_only = 1 if (any(w in low for w in MILD_WORDS) and d_cnt == 0 and p_loss == 0) else 0
        TEMPLATE_METADATA[f].append((tmpl, d_cnt, p_loss, m_only))


def fast_inject_typo(text: str, rng) -> str:
    words = text.split()
    for idx, w in enumerate(words):
        if len(w) > 3 and rng.random() < 0.08:
            ch_idx = rng.integers(1, len(w) - 1)
            char = chr(rng.integers(97, 122))
            words[idx] = w[:ch_idx] + char + w[ch_idx + 1:]
    return " ".join(words)


def generate_all():
    # 1. Branches
    branches_df = pd.DataFrame([
        {"branch_id": "NCR", "city": "Delhi NCR", "n_technicians": 9},
        {"branch_id": "PUN", "city": "Pune", "n_technicians": 6},
        {"branch_id": "JAI", "city": "Jaipur", "n_technicians": 5}
    ])
    branches_df.sort_values("branch_id").to_csv(OUT_DIR / "branches.csv", index=False)

    # 2. Technicians
    tech_data = []
    t_id = 1
    for b_id, count in [("NCR", 9), ("PUN", 6), ("JAI", 5)]:
        for _ in range(count):
            tech_data.append({
                "tech_id": f"T-{t_id:03d}",
                "branch_id": b_id,
                "experience_years": int(RNG.integers(1, 16)),
                "skill_level": int(RNG.integers(1, 4))
            })
            t_id += 1
    technicians_df = pd.DataFrame(tech_data)
    technicians_df.sort_values("tech_id").to_csv(OUT_DIR / "technicians.csv", index=False)
    branch_techs = technicians_df.groupby("branch_id")["tech_id"].apply(list).to_dict()
    tech_exp_map = technicians_df.set_index("tech_id")["experience_years"].to_dict()

    # 3. Weather
    date_range = pd.date_range(DATA_START, DATA_END, freq="D")
    weather_data = []
    festivals = [
        pd.Timestamp("2024-03-25"), pd.Timestamp("2024-11-01"),
        pd.Timestamp("2025-03-14"), pd.Timestamp("2025-10-20")
    ]
    base_temps = {"NCR": 25.0, "PUN": 26.0, "JAI": 26.0}

    for dt in date_range:
        doy = dt.dayofyear
        is_fest = int(any(abs((dt - fest).days) <= 1 for fest in festivals))
        is_monsoon = (dt.month >= 7) and (dt.month <= 9)

        for b in BRANCHES:
            temp = base_temps[b] + 10.0 * np.sin(2.0 * np.pi * (doy - 105) / 365.0) + RNG.normal(0, 2.0)
            if is_monsoon:
                rain = float(np.clip(RNG.exponential(18.0) + RNG.uniform(5.0, 35.0), 0.0, 180.0))
            else:
                rain = float(np.clip(RNG.exponential(1.5) if RNG.random() < 0.25 else 0.0, 0.0, 30.0))

            weather_data.append({
                "date": dt.strftime("%Y-%m-%d"),
                "branch_id": b,
                "temp_c": round(float(temp), 1),
                "rain_mm": round(float(rain), 1),
                "festival_flag": is_fest
            })

    weather_df = pd.DataFrame(weather_data)
    weather_df.sort_values(["branch_id", "date"]).to_csv(OUT_DIR / "weather.csv", index=False)

    # 4. Customers
    n_customers = 18000
    p_weights = [0.55, 0.30, 0.15]
    b_weights = [0.44, 0.31, 0.25]
    cust_branches = RNG.choice(BRANCHES, size=n_customers, p=b_weights)
    cust_premises = RNG.choice(PREMISES, size=n_customers, p=p_weights)
    has_missing_age = RNG.random(n_customers) < 0.02
    raw_ages = np.clip(np.round(RNG.normal(18, 10, n_customers)), 0, 50)
    s_days = RNG.integers(0, 730, n_customers)

    cust_data = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n_customers):
        age = np.nan if has_missing_age[i] else int(raw_ages[i])
        s_date = (base_date + pd.Timedelta(days=int(s_days[i]))).strftime("%Y-%m-%d")
        cust_data.append({
            "customer_id": f"C-{i+1:05d}",
            "branch_id": cust_branches[i],
            "premises_type": cust_premises[i],
            "house_age_years": age,
            "signup_date": s_date
        })

    customers_df = pd.DataFrame(cust_data)
    customers_df.sort_values("customer_id").to_csv(OUT_DIR / "customers.csv", index=False)

    # 5. Rate Card Precomputation
    base_hours = {"wiring": 2.6, "mcb": 1.4, "ac": 2.2, "geyser": 1.6, "fan": 1.1, "inverter": 1.8, "light": 0.8}
    premises_factor = {"apartment": 1.0, "house": 1.15, "shop_office": 1.25}

    rate_card_rows = []
    for f in FAULTS:
        for p in PREMISES:
            expected_parts_cost = sum(prob * PART_UNIT_PRICES[part] for part, prob in FAULT_PART_PROBS[f])
            sim_dur = base_hours[f] * (1.0 + 0.012 * 8.0) * (1.2 - 0.035 * 7.0) * premises_factor[p]
            sim_cost = 380.0 * sim_dur + expected_parts_cost + 199.0
            base_p = float(round(1.55 * sim_cost / 10.0) * 10)
            rate_card_rows.append({"fault_category": f, "premises_type": p, "base_price_inr": base_p})
    rate_card_df = pd.DataFrame(rate_card_rows)
    rate_card_df.sort_values(["fault_category", "premises_type"]).to_csv(OUT_DIR / "rate_card.csv", index=False)
    rate_card_lookup = rate_card_df.set_index(["fault_category", "premises_type"])["base_price_inr"].to_dict()

    # 6. Pre-Sampling for High-Speed Leads & Jobs Generation
    total_target_leads = 65000

    sample_probs = np.ones(n_customers)
    repeat_indices = RNG.choice(n_customers, size=int(0.30 * n_customers), replace=False)
    sample_probs[repeat_indices] = 4.0
    sample_probs /= sample_probs.sum()

    lead_cust_idxs = RNG.choice(n_customers, size=total_target_leads, p=sample_probs)

    start_epoch = int(pd.Timestamp("2024-01-01").timestamp())
    end_epoch = int(pd.Timestamp("2025-12-31 23:59:59").timestamp())
    lead_epochs = RNG.integers(start_epoch, end_epoch, size=total_target_leads)

    lead_channels = RNG.choice(CHANNELS, size=total_target_leads, p=[0.30, 0.20, 0.20, 0.20, 0.10])
    lead_resps = np.clip(RNG.lognormal(mean=np.log(30), sigma=0.8, size=total_target_leads), 2, 300).astype(int)
    lead_quote_ratios = np.clip(RNG.normal(1.10, 0.20, size=total_target_leads), 0.70, 1.80)
    lead_rand_app = RNG.random(size=total_target_leads)
    lead_norm_urg = RNG.normal(0, 0.3, size=total_target_leads)
    lead_bern_book = RNG.random(size=total_target_leads)

    fault_base_prob = {"wiring": 0.16, "mcb": 0.18, "ac": 0.17, "geyser": 0.12, "fan": 0.13, "inverter": 0.10, "light": 0.14}
    channel_mults = {"google_ads": 0.9, "referral": 1.2, "website": 1.0, "whatsapp": 1.1, "walk_in": 1.3}
    base_urg_map = {"wiring": 2.0, "mcb": 1.2, "ac": 0.8, "geyser": 1.0, "fan": 0.3, "inverter": 0.8, "light": 0.2}
    app_map = {
        "ac": "ac", "geyser": "geyser", "fan": "fan_cooler",
        "inverter": "inverter_ups", "mcb": "switchboard",
        "wiring": "switchboard", "light": "lights_switches"
    }

    month_fault_probs = {}
    for m in range(1, 13):
        f_probs = fault_base_prob.copy()
        if 4 <= m <= 6:
            f_probs["ac"] *= 2.2
            f_probs["fan"] *= 2.2
        elif m in [12, 1, 2]:
            f_probs["geyser"] *= 2.0
        elif 7 <= m <= 9:
            f_probs["wiring"] *= 1.6
            f_probs["mcb"] *= 1.6
        tot = sum(f_probs.values())
        keys = list(f_probs.keys())
        probs = [f_probs[k] / tot for k in keys]
        month_fault_probs[m] = (keys, probs)

    lead_rows = []
    job_rows = []
    job_id_ctr = 1

    for i in range(total_target_leads):
        c_idx = lead_cust_idxs[i]
        cust = cust_data[c_idx]

        dt = datetime.datetime.fromtimestamp(int(lead_epochs[i]), tz=datetime.timezone.utc)
        m = dt.month

        f_keys, f_probs = month_fault_probs[m]
        fault = RNG.choice(f_keys, p=f_probs)

        rapp = lead_rand_app[i]
        if rapp < 0.15:
            appliance = "unknown"
        elif rapp < 0.20:
            appliance = APPLIANCES[RNG.integers(0, len(APPLIANCES))]
        else:
            appliance = app_map[fault]

        tmpl_idx = RNG.integers(0, len(TEMPLATE_METADATA[fault]))
        base_text, d_cnt, p_loss, m_only = TEMPLATE_METADATA[fault][tmpl_idx]
        complaint = fast_inject_typo(base_text, RNG)

        urg_score = (
            base_urg_map[fault] +
            1.5 * d_cnt +
            (0.5 if cust["premises_type"] == "shop_office" else 0.0) +
            (0.5 * p_loss) -
            (0.8 * m_only) +
            lead_norm_urg[i]
        )
        urgency = "high" if urg_score >= 2.5 else ("medium" if urg_score >= 1.0 else "low")

        channel = lead_channels[i]
        resp_time = int(lead_resps[i])
        resp_f = 0.6 + 0.4 * np.exp(-resp_time / 70.0)

        base_rate = rate_card_lookup[(fault, cust["premises_type"])]
        typical_p = float(base_rate * (1.12 if urgency == "high" else 1.0))
        q_ratio = float(lead_quote_ratios[i])
        quoted_p = round(typical_p * q_ratio, 2)

        k = 4.2 if urgency == "high" else 6.0
        p_book = np.clip(
            (channel_mults[channel] * resp_f) / (1.0 + np.exp(k * (q_ratio - 1.08))),
            0.0, 1.0
        )
        booked = 1 if lead_bern_book[i] < p_book else 0

        lid = f"L-{i+1:06d}"
        lead_dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")

        lead_rows.append({
            "lead_id": lid,
            "created_at": lead_dt_str,
            "customer_id": cust["customer_id"],
            "branch_id": cust["branch_id"],
            "source_channel": channel,
            "complaint_text": complaint,
            "appliance_type": appliance,
            "premises_type": cust["premises_type"],
            "house_age_years": cust["house_age_years"],
            "fault_category": fault,
            "urgency": urgency,
            "response_time_min": resp_time,
            "typical_price": round(typical_p, 2),
            "quoted_price": quoted_p,
            "quote_ratio": round(quoted_p / typical_p, 4),
            "booked": booked
        })

        if booked == 1:
            jid = f"J-{job_id_ctr:06d}"
            job_id_ctr += 1

            day_lag = RNG.integers(0, 2) if urgency == "high" else RNG.integers(0, 4)
            j_dt = dt + datetime.timedelta(days=int(day_lag))
            j_date_str = min(j_dt.strftime("%Y-%m-%d"), DATA_END)

            tech = branch_techs[cust["branch_id"]][RNG.integers(0, len(branch_techs[cust["branch_id"]]))]
            t_exp = tech_exp_map[tech]

            h_age = 18 if np.isnan(cust["house_age_years"]) else cust["house_age_years"]
            age_f = 1.0 + 0.012 * max(0.0, float(h_age - 10))
            exp_f = 1.2 - 0.035 * min(float(t_exp), 10.0)
            prem_f = premises_factor[cust["premises_type"]]

            dur = base_hours[fault] * age_f * exp_f * prem_f * float(RNG.lognormal(0.0, 0.15))
            if RNG.random() < 0.01:
                dur *= 3.0
            dur = round(float(dur), 2)

            assigned_parts = []
            for p_name, p_prob in FAULT_PART_PROBS[fault]:
                if RNG.random() < p_prob:
                    assigned_parts.append(p_name)
            parts_str = "|".join(assigned_parts)
            n_parts = len(assigned_parts)
            parts_cost = sum(PART_UNIT_PRICES[p] for p in assigned_parts)
            j_cost = round(380.0 * dur + parts_cost + 199.0, 2)

            p_rep = 0.04 + (0.10 if t_exp < 3 else 0.0) + (0.05 if n_parts >= 3 else 0.0)
            rep_visit = 1 if RNG.random() < p_rep else 0

            job_rows.append({
                "job_id": jid,
                "lead_id": lid,
                "customer_id": cust["customer_id"],
                "tech_id": tech,
                "branch_id": cust["branch_id"],
                "job_date": j_date_str,
                "created_hour": dt.hour,
                "day_of_week": dt.weekday(),
                "complaint_text": complaint,
                "appliance_type": appliance,
                "premises_type": cust["premises_type"],
                "house_age_years": cust["house_age_years"],
                "fault_category": fault,
                "urgency": urgency,
                "parts_used": parts_str,
                "n_parts": n_parts,
                "duration_hours": dur,
                "job_cost_inr": j_cost,
                "final_price_inr": quoted_p,
                "repeat_visit": rep_visit
            })

    leads_df = pd.DataFrame(lead_rows)
    leads_df.sort_values("lead_id").to_csv(OUT_DIR / "leads.csv", index=False)

    jobs_df = pd.DataFrame(job_rows)
    jobs_df.sort_values("job_id").to_csv(OUT_DIR / "jobs.csv", index=False)

    # 7. Checksums
    checksum_lines = []
    for name in ["branches", "technicians", "customers", "rate_card", "leads", "jobs", "weather"]:
        p = OUT_DIR / f"{name}.csv"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        checksum_lines.append(f"{sha}  {name}.csv")

    (OUT_DIR / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Data generation complete: {len(jobs_df):,} jobs, {len(leads_df):,} leads.")


if __name__ == "__main__":
    generate_all()
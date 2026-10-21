"""Synthetic data generator implementing Voltline Contract v1.0 specifications."""

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
    "led_driver": 250
}

FAULT_PART_PROBS = {
    "wiring": {"copper_wire": 0.80, "connectors_tape": 0.70, "switch_socket": 0.45, "mcb": 0.25},
    "mcb": {"mcb": 0.70, "rccb": 0.30, "cable_lugs": 0.50},
    "ac": {"run_capacitor": 0.55, "contactor": 0.30, "isolator": 0.35},
    "geyser": {"thermostat": 0.60, "heating_element": 0.45, "geyser_plug": 0.30},
    "fan": {"fan_capacitor": 0.60, "regulator": 0.40, "bearing_set": 0.30},
    "inverter": {"battery_terminal": 0.40, "fuse": 0.50},
    "light": {"switch_socket": 0.60, "connectors_tape": 0.40, "led_driver": 0.30}
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


def inject_typo(text: str) -> str:
    """Randomly swaps or drops one character with 8% probability per word."""
    words = text.split()
    new_words = []
    for w in words:
        if len(w) > 3 and RNG.random() < 0.08:
            idx = RNG.integers(1, len(w) - 1)
            char = chr(RNG.integers(97, 122))
            w = w[:idx] + char + w[idx + 1:]
        new_words.append(w)
    return " ".join(new_words)


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

    # 4. Weekly Job Demand Distribution to hit target ~26,500 jobs
    base_jobs = {"NCR": 112, "PUN": 78, "JAI": 64}
    season_mult = {
        1: 0.85, 2: 0.85, 3: 0.95, 4: 1.10, 5: 1.20, 6: 1.15,
        7: 1.10, 8: 1.10, 9: 1.00, 10: 0.95, 11: 1.05, 12: 0.90
    }

    weekly_demand = []
    cur_monday = pd.Timestamp("2024-01-01")
    while cur_monday <= pd.Timestamp(DATA_END):
        week_fest = any(abs((cur_monday + pd.Timedelta(days=d) - fest).days) <= 1
                        for d in range(7) for fest in festivals)
        f_mult = 1.2 if week_fest else 1.0
        m_mult = season_mult[cur_monday.month]

        for b in BRANCHES:
            lam = base_jobs[b] * m_mult * f_mult
            n_jobs = RNG.poisson(lam)
            weekly_demand.append((cur_monday, b, n_jobs))
        cur_monday += pd.Timedelta(days=7)

    # Convert demand to jobs target per branch & day
    total_target_jobs = sum(x[2] for x in weekly_demand)
    total_target_leads = int(total_target_jobs / 0.375)

    # 5. Customers
    # 70% 1 job, 30% 2-8 jobs
    n_customers = int(total_target_jobs * 0.72)
    cust_data = []
    p_weights = [0.55, 0.30, 0.15]
    b_weights = [0.44, 0.31, 0.25]

    for c_idx in range(1, n_customers + 1):
        b = RNG.choice(BRANCHES, p=b_weights)
        prem = RNG.choice(PREMISES, p=p_weights)
        if RNG.random() < 0.02:
            age = np.nan
        else:
            age = int(np.clip(round(RNG.normal(18, 10)), 0, 50))
        s_day = RNG.integers(0, 730)
        s_date = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(s_day))).strftime("%Y-%m-%d")

        cust_data.append({
            "customer_id": f"C-{c_idx:05d}",
            "branch_id": b,
            "premises_type": prem,
            "house_age_years": age,
            "signup_date": s_date
        })

    customers_df = pd.DataFrame(cust_data)
    customers_df.sort_values("customer_id").to_csv(OUT_DIR / "customers.csv", index=False)
    cust_dict = {c["customer_id"]: c for c in cust_data}
    branch_techs = technicians_df.groupby("branch_id")["tech_id"].apply(list).to_dict()
    tech_exp_map = technicians_df.set_index("tech_id")["experience_years"].to_dict()

    # 6. Rate Card Precomputation
    base_hours = {"wiring": 2.6, "mcb": 1.4, "ac": 2.2, "geyser": 1.6, "fan": 1.1, "inverter": 1.8, "light": 0.8}
    premises_factor = {"apartment": 1.0, "house": 1.15, "shop_office": 1.25}

    rate_card_rows = []
    for f in FAULTS:
        for p in PREMISES:
            expected_parts_cost = sum(
                prob * PART_UNIT_PRICES[part]
                for part, prob in FAULT_PART_PROBS[f].items()
            )
            sim_dur = base_hours[f] * (1.0 + 0.012 * 8.0) * (1.2 - 0.035 * 7.0) * premises_factor[p]
            sim_cost = 380.0 * sim_dur + expected_parts_cost + 199.0
            base_p = float(round(1.55 * sim_cost / 10.0) * 10)
            rate_card_rows.append({
                "fault_category": f,
                "premises_type": p,
                "base_price_inr": base_p
            })
    rate_card_df = pd.DataFrame(rate_card_rows)
    rate_card_df.sort_values(["fault_category", "premises_type"]).to_csv(OUT_DIR / "rate_card.csv", index=False)
    rate_card_lookup = rate_card_df.set_index(["fault_category", "premises_type"])["base_price_inr"].to_dict()

    # 7. Generate Leads & Jobs
    lead_rows = []
    job_rows = []

    channel_probs = [0.30, 0.20, 0.20, 0.20, 0.10]
    channel_mults = {"google_ads": 0.9, "referral": 1.2, "website": 1.0, "whatsapp": 1.1, "walk_in": 1.3}
    fault_base_prob = {"wiring": 0.16, "mcb": 0.18, "ac": 0.17, "geyser": 0.12, "fan": 0.13, "inverter": 0.10, "light": 0.14}

    cust_ids = list(cust_dict.keys())
    # 70% 1 job, 30% multiple (weighted selection)
    sample_probs = np.ones(len(cust_ids))
    repeat_indices = RNG.choice(len(cust_ids), size=int(0.30 * len(cust_ids)), replace=False)
    sample_probs[repeat_indices] = 4.0
    sample_probs /= sample_probs.sum()

    start_ts = pd.Timestamp(DATA_START).timestamp()
    end_ts = pd.Timestamp(DATA_END).timestamp()

    lead_id_ctr = 1
    job_id_ctr = 1

    for _ in range(total_target_leads):
        cid = RNG.choice(cust_ids, p=sample_probs)
        cust = cust_dict[cid]

        ts = RNG.uniform(start_ts, end_ts)
        lead_dt = pd.to_datetime(ts, unit="s")
        month = lead_dt.month

        # Month seasonal shifts
        f_probs = fault_base_prob.copy()
        if 4 <= month <= 6:
            f_probs["ac"] *= 2.2
            f_probs["fan"] *= 2.2
        elif month in [12, 1, 2]:
            f_probs["geyser"] *= 2.0
        elif 7 <= month <= 9:
            f_probs["wiring"] *= 1.6
            f_probs["mcb"] *= 1.6

        f_total = sum(f_probs.values())
        f_keys = list(f_probs.keys())
        f_p = [f_probs[k] / f_total for k in f_keys]
        fault = RNG.choice(f_keys, p=f_p)

        # Appliance
        rand_val = RNG.random()
        if rand_val < 0.15:
            appliance = "unknown"
        elif rand_val < 0.20:
            appliance = RNG.choice(APPLIANCES)
        else:
            mapping = {
                "ac": "ac", "geyser": "geyser", "fan": "fan_cooler",
                "inverter": "inverter_ups", "mcb": "switchboard",
                "wiring": "switchboard", "light": "lights_switches"
            }
            appliance = mapping[fault]

        # Complaint text
        base_tmpl = RNG.choice(COMPLAINT_TEMPLATES[fault])
        complaint = inject_typo(base_tmpl)
        lower_comp = complaint.lower()

        # Urgency score calculation
        base_urg = {"wiring": 2.0, "mcb": 1.2, "ac": 0.8, "geyser": 1.0, "fan": 0.3, "inverter": 0.8, "light": 0.2}[fault]
        danger_count = sum(1 for w in DANGER_WORDS if w in lower_comp)
        power_loss = 1 if any(w in lower_comp for w in POWER_LOSS_WORDS) else 0
        mild_only = 1 if (any(w in lower_comp for w in MILD_WORDS) and danger_count == 0 and power_loss == 0) else 0

        urg_score = (
            base_urg +
            1.5 * danger_count +
            (0.5 if cust["premises_type"] == "shop_office" else 0.0) +
            (0.5 * power_loss) -
            (0.8 * mild_only) +
            RNG.normal(0, 0.3)
        )
        if urg_score >= 2.5:
            urgency = "high"
        elif urg_score >= 1.0:
            urgency = "medium"
        else:
            urgency = "low"

        # Channel & Booking
        channel = RNG.choice(CHANNELS, p=channel_probs)
        resp_time = int(np.clip(RNG.lognormal(mean=np.log(30), sigma=0.8), 2, 300))
        resp_f = 0.6 + 0.4 * np.exp(-resp_time / 70.0)

        base_rate = rate_card_lookup[(fault, cust["premises_type"])]
        typical_p = float(base_rate * (1.12 if urgency == "high" else 1.0))
        quote_ratio = float(np.clip(RNG.normal(1.10, 0.20), 0.70, 1.80))
        quoted_p = round(typical_p * quote_ratio, 2)

        k = 4.2 if urgency == "high" else 6.0
        p_book = float(np.clip(
            (channel_mults[channel] * resp_f) / (1.0 + np.exp(k * (quote_ratio - 1.08))),
            0.0, 1.0
        ))
        booked = int(RNG.binomial(1, p_book))

        lid = f"L-{lead_id_ctr:06d}"
        lead_id_ctr += 1

        lead_rows.append({
            "lead_id": lid,
            "created_at": lead_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "customer_id": cid,
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

            day_lag = int(RNG.integers(0, 2) if urgency == "high" else RNG.integers(0, 4))
            j_date = lead_dt + pd.Timedelta(days=day_lag)
            if j_date > pd.Timestamp(DATA_END):
                j_date = pd.Timestamp(DATA_END)

            tech = RNG.choice(branch_techs[cust["branch_id"]])
            t_exp = tech_exp_map[tech]

            # House age factor
            h_age = 18 if np.isnan(cust["house_age_years"]) else cust["house_age_years"]
            age_f = 1.0 + 0.012 * max(0.0, float(h_age - 10))
            exp_f = 1.2 - 0.035 * min(float(t_exp), 10.0)
            prem_f = premises_factor[cust["premises_type"]]

            dur = (
                base_hours[fault] * age_f * exp_f * prem_f *
                float(RNG.lognormal(0.0, 0.15))
            )
            if RNG.random() < 0.01:
                dur *= 3.0
            dur = round(float(dur), 2)

            # Parts used
            assigned_parts = []
            for p_name, p_prob in FAULT_PART_PROBS[fault].items():
                if RNG.random() < p_prob:
                    assigned_parts.append(p_name)
            parts_str = "|".join(assigned_parts)
            n_parts = len(assigned_parts)
            parts_cost = sum(PART_UNIT_PRICES[p] for p in assigned_parts)

            j_cost = round(380.0 * dur + parts_cost + 199.0, 2)

            # Repeat probability
            p_rep = 0.04 + (0.10 if t_exp < 3 else 0.0) + (0.05 if n_parts >= 3 else 0.0)
            rep_visit = int(RNG.binomial(1, np.clip(p_rep, 0.0, 1.0)))

            job_rows.append({
                "job_id": jid,
                "lead_id": lid,
                "customer_id": cid,
                "tech_id": tech,
                "branch_id": cust["branch_id"],
                "job_date": j_date.strftime("%Y-%m-%d"),
                "created_hour": lead_dt.hour,
                "day_of_week": lead_dt.weekday(),
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

    # 8. SHA-256 Checksums
    checksum_lines = []
    csv_names = ["branches", "technicians", "customers", "rate_card", "leads", "jobs", "weather"]
    for name in csv_names:
        p = OUT_DIR / f"{name}.csv"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        checksum_lines.append(f"{sha}  {name}.csv")

    (OUT_DIR / "CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    # 9. Data Dictionary
    data_dict_content = """# Voltline Data Dictionary (v1.0)

### branches.csv
- `branch_id`: Primary key (NCR, PUN, JAI).
- `city`: Operating city hub.
- `n_technicians`: Number of full-time electrical field technicians.

### technicians.csv
- `tech_id`: Primary key (`T-xxx`).
- `branch_id`: Foreign key -> branches.
- `experience_years`: Technician total experience (1-15).
- `skill_level`: Competency tier (1=junior, 2=standard, 3=expert).

### customers.csv
- `customer_id`: Primary key (`C-xxxxx`).
- `branch_id`: Operating branch.
- `premises_type`: Property category (apartment, house, shop_office).
- `house_age_years`: Structure age (0-50, ~2% missing as NaN).
- `signup_date`: Date registered.

### rate_card.csv
- `fault_category`: Primary electrical defect category.
- `premises_type`: Property category.
- `base_price_inr`: Standard baseline charge before urgency markup.

### leads.csv
- `lead_id`: Primary key (`L-xxxxxx`).
- `created_at`: Inquiry timestamp.
- `customer_id`: Foreign key -> customers.
- `branch_id`: Foreign key -> branches.
- `source_channel`: Channel (google_ads, referral, website, whatsapp, walk_in).
- `complaint_text`: Raw inquiry text in English/Hinglish with realistic typos.
- `appliance_type`: Stated device category or 'unknown'.
- `premises_type`: Property structure category.
- `house_age_years`: Structure age in years.
- `fault_category`: True electrical domain fault.
- `urgency`: Dispatch severity rating (low, medium, high).
- `response_time_min`: First response latency in minutes.
- `typical_price`: Market reference rate.
- `quoted_price`: Proposed customer charge.
- `quote_ratio`: quoted_price / typical_price.
- `booked`: Lead conversion flag (1 = converted to job, 0 = lost).

### jobs.csv
- `job_id`: Primary key (`J-xxxxxx`). One job per booked lead.
- `lead_id`: Originating lead ID.
- `customer_id`: Foreign key -> customers.
- `tech_id`: Assigned technician ID.
- `branch_id`: Branch ID.
- `job_date`: Service execution date.
- `created_hour`: Time of lead creation (0-23).
- `day_of_week`: Day index (0 = Monday).
- `complaint_text`: Inquiry text.
- `appliance_type`, `premises_type`, `house_age_years`, `fault_category`, `urgency`: Synced from lead.
- `parts_used`: Pipe-delimited list of replaced parts.
- `n_parts`: Total parts replaced.
- `duration_hours`: On-site execution time.
- `job_cost_inr`: Cost of delivery: ₹380/hr + parts sum + ₹199 base.
- `final_price_inr`: Billed revenue (matches quoted_price).
- `repeat_visit`: Service call-back flag within 30 days (0 or 1).

### weather.csv
- `date`: Calendar date.
- `branch_id`: City code.
- `temp_c`: Average daily ambient temperature.
- `rain_mm`: Precipitation in millimeters.
- `festival_flag`: 1 if within +/- 1 day of a major festive season.

### Fixed Part Unit Prices (INR)
| Part | Price | Part | Price |
|---|---|---|---|
| copper_wire | 420 | thermostat | 350 |
| connectors_tape | 90 | heating_element | 700 |
| switch_socket | 180 | geyser_plug | 150 |
| mcb | 380 | fan_capacitor | 60 |
| rccb | 1100 | regulator | 220 |
| cable_lugs | 60 | bearing_set | 180 |
| run_capacitor | 520 | battery_terminal | 200 |
| contactor | 650 | fuse | 50 |
| isolator | 300 | led_driver | 250 |
"""
    (OUT_DIR / "data_dictionary.md").write_text(data_dict_content, encoding="utf-8")
    print(f"Dataset generated successfully:\n - Jobs: {len(jobs_df):,}\n - Leads: {len(leads_df):,}\n - Customers: {len(customers_df):,}")


if __name__ == "__main__":
    generate_all()
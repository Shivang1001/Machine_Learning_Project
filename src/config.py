"""Shared configuration and domain constants across all Voltline modules."""

RANDOM_STATE = 42
DATA_START = "2024-01-01"
DATA_END = "2025-12-31"
TRAIN_END = "2025-06-30"
VAL_END = "2025-09-30"

FAULTS = ["wiring", "mcb", "ac", "geyser", "fan", "inverter", "light"]
URGENCY = ["low", "medium", "high"]
APPLIANCES = ["ac", "geyser", "fan_cooler", "inverter_ups", "switchboard", "lights_switches", "unknown"]
PREMISES = ["apartment", "house", "shop_office"]
CHANNELS = ["google_ads", "referral", "website", "whatsapp", "walk_in"]
BRANCHES = ["NCR", "PUN", "JAI"]
PARTS = [
    "copper_wire", "connectors_tape", "switch_socket", "mcb", "rccb",
    "cable_lugs", "run_capacitor", "contactor", "isolator", "thermostat",
    "heating_element", "geyser_plug", "fan_capacitor", "regulator",
    "bearing_set", "battery_terminal", "fuse", "led_driver"
]
JOBS_PER_TECH_PER_WEEK = 13
# Voltline Data Dictionary (v1.0)

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

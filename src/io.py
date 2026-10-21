"""Standardized data loading and time-based splitting routines."""

from pathlib import Path
from typing import Tuple
import pandas as pd
from src.config import TRAIN_END, VAL_END

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

DATE_COLS = {
    "customers": ["signup_date"],
    "leads": ["created_at"],
    "jobs": ["job_date"],
    "weather": ["date"]
}


def load(name: str) -> pd.DataFrame:
    """Loads a processed CSV file by name with correct date parsing."""
    valid_names = {"branches", "technicians", "customers", "leads", "jobs", "weather", "rate_card"}
    if name not in valid_names:
        raise ValueError(f"Unknown dataset '{name}'. Expected one of {sorted(valid_names)}")

    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}. Run generator first.")

    parse_dates = DATE_COLS.get(name, False)
    return pd.read_csv(path, parse_dates=parse_dates)


def split(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a DataFrame chronologically into train, val, and test subsets."""
    train_cutoff = pd.to_datetime(TRAIN_END)
    val_cutoff = pd.to_datetime(VAL_END)

    dates = pd.to_datetime(df[date_col])
    train = df[dates <= train_cutoff].copy()
    val = df[(dates > train_cutoff) & (dates <= val_cutoff)].copy()
    test = df[dates > val_cutoff].copy()
    return train, val, test
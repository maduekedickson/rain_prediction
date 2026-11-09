"""
Data ingestion and preprocessing module for Rainfall Prediction System.
Handles schema normalization, column sanitization, and data validation.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

# Core meteorological feature schema expected by the model
FEATURE_COLUMNS: List[str] = [
    "pressure",
    "maxtemp",
    "mintemp",
    "dewpoint",
    "humidity",
    "cloud",
    "sunshine",
    "winddirection",
    "windspeed",
]

TARGET_COLUMN: str = "rainfall"


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans leading/trailing whitespaces and standardizes column aliases.
    """
    df = df.copy()
    # Strip whitespace and lowercase all column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Handle known column aliases
    alias_map = {
        "temparature": "temperature",
        "wind_direction": "winddirection",
        "wind_speed": "windspeed",
        "max_temp": "maxtemp",
        "min_temp": "mintemp",
        "dew_point": "dewpoint",
    }
    df = df.rename(columns=alias_map)
    return df


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """
    Loads raw CSV and applies schema sanitation.
    """
    df = pd.read_csv(csv_path)
    df = sanitize_column_names(df)
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extracts validated feature matrix X and target series y (if target exists).
    """
    df_clean = sanitize_column_names(df)

    # Encode binary target if present
    y: Optional[pd.Series] = None
    if TARGET_COLUMN in df_clean.columns:
        target_series = df_clean[TARGET_COLUMN].copy()
        if target_series.dtype == object:
            # Map strings like 'yes'/'no' or '1'/'0'
            y = target_series.astype(str).str.strip().str.lower().map({
                "yes": 1,
                "y": 1,
                "1": 1,
                "true": 1,
                "no": 0,
                "n": 0,
                "0": 0,
                "false": 0,
            })
        else:
            y = target_series.astype(int)

    # Extract required feature columns
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_clean.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required feature columns: {missing_cols}")

    X = df_clean[FEATURE_COLUMNS].copy()
    return X, y

"""
Inference service for Rainfall Prediction.
Wraps loaded scikit-learn pipeline to compute continuous probabilities,
categorical risk tiers, and meteorological interpretations.
"""

from typing import Dict, Any, Union
import numpy as np

FEATURE_COLUMNS = [
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


def classify_risk_tier(probability: float) -> Dict[str, str]:
    """
    Categorizes probability of precipitation (PoP) into actionable risk tiers.
    """
    if probability < 0.30:
        return {
            "tier": "Low Risk",
            "badge_color": "green",
            "message": "Dry weather expected. Minimal chance of precipitation.",
            "icon": "☀️",
        }
    elif probability < 0.70:
        return {
            "tier": "Moderate Risk",
            "badge_color": "orange",
            "message": "Precipitation possible. Consider carrying rain gear.",
            "icon": "⛅",
        }
    else:
        return {
            "tier": "High Risk",
            "badge_color": "red",
            "message": "High likelihood of rainfall. Precipitation expected.",
            "icon": "🌧️",
        }


def predict_single(
    pipeline: Any,
    input_features: Dict[str, Union[float, int]],
) -> Dict[str, Any]:
    """
    Performs inference for a single day's observations.
    """
    import pandas as pd

    # Ensure all required features are present
    row_data = {}
    for col in FEATURE_COLUMNS:
        if col not in input_features:
            raise ValueError(f"Missing required meteorological parameter: '{col}'")
        row_data[col] = float(input_features[col])

    df_in = pd.DataFrame([row_data], columns=FEATURE_COLUMNS)

    # Compute binary prediction and class probability
    pred_class = int(pipeline.predict(df_in)[0])
    probabilities = pipeline.predict_proba(df_in)[0]
    prob_rain = float(probabilities[1])

    risk_info = classify_risk_tier(prob_rain)

    return {
        "prediction": pred_class,
        "prediction_label": "Rain" if pred_class == 1 else "No Rain",
        "probability_rain": prob_rain,
        "probability_dry": float(probabilities[0]),
        "risk_tier": risk_info["tier"],
        "icon": risk_info["icon"],
        "message": risk_info["message"],
    }


def predict_batch(
    pipeline: Any,
    df_batch: Any,
) -> Any:
    """
    Performs inference for a batch dataframe and appends prediction columns.
    """
    from src.data_loader import sanitize_column_names

    df_clean = sanitize_column_names(df_batch)

    missing = [c for c in FEATURE_COLUMNS if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Uploaded batch CSV is missing required columns: {missing}")

    X = df_clean[FEATURE_COLUMNS].copy()
    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    result_df = df_batch.copy()
    result_df["Predicted_Rainfall"] = np.where(preds == 1, "Rain", "No Rain")
    result_df["Rain_Probability (%)"] = np.round(probs * 100, 1)
    result_df["Risk_Level"] = [classify_risk_tier(p)["tier"] for p in probs]

    return result_df

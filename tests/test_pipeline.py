"""
Unit tests for data loader, feature preprocessing, and inference pipeline.
Engineered to run seamlessly across local environments and GitHub Actions CI.
"""

import pytest
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except (ImportError, Exception):
    PANDAS_AVAILABLE = False

from src.predict import classify_risk_tier, FEATURE_COLUMNS


def test_classify_risk_tier():
    low = classify_risk_tier(0.15)
    assert low["tier"] == "Low Risk"
    assert low["icon"] == "☀️"

    med = classify_risk_tier(0.45)
    assert med["tier"] == "Moderate Risk"
    assert med["icon"] == "⛅"

    high = classify_risk_tier(0.85)
    assert high["tier"] == "High Risk"
    assert high["icon"] == "🌧️"


def test_classify_risk_tier_boundary():
    assert classify_risk_tier(0.0)["tier"] == "Low Risk"
    assert classify_risk_tier(0.299)["tier"] == "Low Risk"
    assert classify_risk_tier(0.30)["tier"] == "Moderate Risk"
    assert classify_risk_tier(0.699)["tier"] == "Moderate Risk"
    assert classify_risk_tier(0.70)["tier"] == "High Risk"
    assert classify_risk_tier(1.0)["tier"] == "High Risk"


def test_feature_columns_contract():
    expected_features = [
        "pressure", "maxtemp", "mintemp", "dewpoint",
        "humidity", "cloud", "sunshine", "winddirection", "windspeed"
    ]
    assert FEATURE_COLUMNS == expected_features
    assert len(FEATURE_COLUMNS) == 9


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas blocked by local OS policy; runs on CI")
def test_sanitize_column_names():
    from src.data_loader import sanitize_column_names
    dirty_df = pd.DataFrame({
        "pressure ": [1012.0],
        "humidity ": [85],
        "         winddirection": [180],
        "temparature": [24.0],
    })
    cleaned = sanitize_column_names(dirty_df)
    assert "pressure" in cleaned.columns
    assert "humidity" in cleaned.columns
    assert "winddirection" in cleaned.columns
    assert "temperature" in cleaned.columns


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="Pandas blocked by local OS policy; runs on CI")
def test_prepare_features_and_target():
    from src.data_loader import prepare_features_and_target
    data = {col: [1.0, 2.0] for col in FEATURE_COLUMNS}
    data["rainfall"] = ["yes", "no"]
    data["day"] = [1, 2]
    df = pd.DataFrame(data)

    X, y = prepare_features_and_target(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 2
    assert y is not None
    assert list(y.values) == [1, 0]

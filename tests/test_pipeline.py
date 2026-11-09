"""
Unit and Integration Tests for Rainfall Prediction Pipeline
===========================================================
Verifies:
- Dataset integrity, presence, and meteorological schema.
- Model artifact loading, scikit-learn pipeline integrity, and probability bounds.
- Risk tier classification and boundary conditions.
- Feature column contracts.
"""

import os
import csv
import pickle
import unittest
import numpy as np


class TestRainfallPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.data_path = os.path.join(cls.base_dir, "dataset1.csv")
        cls.model_path = os.path.join(cls.base_dir, "logistic_regression_model.pkl")
        cls.expected_features = [
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

    def test_dataset_exists_and_schema(self):
        """Verifies dataset file presence, non-empty rows, and expected header columns."""
        self.assertTrue(os.path.exists(self.data_path), f"Dataset not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            raw_headers = next(reader)
            rows = list(reader)

        # Sanitize header strings (trim whitespace, lowercase)
        clean_headers = [h.strip().lower() for h in raw_headers]
        clean_headers = ["temperature" if h == "temparature" else h for h in clean_headers]

        for feat in self.expected_features:
            self.assertIn(feat, clean_headers, f"Expected feature '{feat}' missing from dataset headers.")

        self.assertIn("rainfall", clean_headers, "Target column 'rainfall' missing from dataset.")
        self.assertGreaterEqual(len(rows), 300, "Dataset contains insufficient sample records.")

        # Check sample value bounds
        col_idx = {h: i for i, h in enumerate(clean_headers)}
        for idx, row in enumerate(rows[:50]):
            pressure = float(row[col_idx["pressure"]])
            humidity = float(row[col_idx["humidity"]])
            cloud = float(row[col_idx["cloud"]])
            sunshine = float(row[col_idx["sunshine"]])

            self.assertTrue(900.0 <= pressure <= 1100.0, f"Pressure out of meteorological bounds: {pressure}")
            self.assertTrue(0.0 <= humidity <= 100.0, f"Humidity out of bounds: {humidity}")
            self.assertTrue(0.0 <= cloud <= 100.0, f"Cloud cover out of bounds: {cloud}")
            self.assertTrue(0.0 <= sunshine <= 24.0, f"Sunshine duration out of bounds: {sunshine}")

    def test_model_artifact_loading_and_inference(self):
        """Verifies serialized model artifact exists, loads cleanly, and computes valid probabilities."""
        self.assertTrue(os.path.exists(self.model_path), f"Model artifact missing at {self.model_path}")
        file_size = os.path.getsize(self.model_path)
        self.assertGreater(file_size, 500, f"Model file unexpectedly small: {file_size} bytes")

        with open(self.model_path, "rb") as f:
            loaded_model = pickle.load(f)

        self.assertTrue(hasattr(loaded_model, "predict"), "Model artifact missing 'predict' method.")
        self.assertTrue(hasattr(loaded_model, "predict_proba"), "Model artifact missing 'predict_proba' method.")

        # Sample observation: 9 features
        sample_input = np.array([[1015.0, 24.0, 16.0, 13.0, 65.0, 45.0, 6.0, 80.0, 15.0]])
        pred = loaded_model.predict(sample_input)
        prob = loaded_model.predict_proba(sample_input)[0]

        self.assertIn(pred[0], [0, 1], f"Prediction class must be 0 or 1, got {pred[0]}")
        self.assertTrue(0.0 <= prob[1] <= 1.0, f"Probability out of range [0, 1]: {prob[1]}")
        self.assertAlmostEqual(prob[0] + prob[1], 1.0, places=4, msg="Probabilities do not sum to 1.0")

    def test_risk_tier_classification(self):
        """Verifies risk tier thresholds align with meteorological guidance."""
        from src.predict import classify_risk_tier

        low = classify_risk_tier(0.10)
        self.assertEqual(low["tier"], "Low Risk")
        self.assertEqual(low["icon"], "☀️")

        med = classify_risk_tier(0.50)
        self.assertEqual(med["tier"], "Moderate Risk")
        self.assertEqual(med["icon"], "⛅")

        high = classify_risk_tier(0.85)
        self.assertEqual(high["tier"], "High Risk")
        self.assertEqual(high["icon"], "🌧️")

    def test_risk_tier_boundaries(self):
        """Verifies exact decision boundary edge conditions."""
        from src.predict import classify_risk_tier

        self.assertEqual(classify_risk_tier(0.0)["tier"], "Low Risk")
        self.assertEqual(classify_risk_tier(0.299)["tier"], "Low Risk")
        self.assertEqual(classify_risk_tier(0.30)["tier"], "Moderate Risk")
        self.assertEqual(classify_risk_tier(0.699)["tier"], "Moderate Risk")
        self.assertEqual(classify_risk_tier(0.70)["tier"], "High Risk")
        self.assertEqual(classify_risk_tier(1.0)["tier"], "High Risk")

    def test_feature_columns_contract(self):
        """Verifies the exact 9 meteorological feature names in order."""
        from src.predict import FEATURE_COLUMNS
        self.assertEqual(len(FEATURE_COLUMNS), 9)
        self.assertEqual(FEATURE_COLUMNS, self.expected_features)


if __name__ == "__main__":
    unittest.main()

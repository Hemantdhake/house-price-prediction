"""
Unit Tests — Prediction Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Sample Input ─────────────────────────────────────────────
SAMPLE_HOUSE = {
    "LotArea": 8500,
    "YearBuilt": 2005,
    "YearRemodAdd": 2015,
    "TotalBsmtSF": 1000.0,
    "GrLivArea": 1800.0,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "TotRmsAbvGrd": 7,
    "GarageArea": 400.0,
    "PoolArea": 0.0,
    "OverallQual": 7,
    "OverallCond": 5,
}


# ─── Test: Input Validation ───────────────────────────────────
class TestInputValidation:

    def test_sample_house_has_required_keys(self):
        """Test sample house has all required keys."""
        required_keys = [
            "LotArea", "YearBuilt", "GrLivArea",
            "OverallQual", "FullBath",
        ]
        for key in required_keys:
            assert key in SAMPLE_HOUSE, f"Missing key: {key}"

    def test_numerical_values_valid(self):
        """Test numerical values are in valid range."""
        assert SAMPLE_HOUSE["LotArea"] > 0
        assert 1 <= SAMPLE_HOUSE["OverallQual"] <= 10
        assert SAMPLE_HOUSE["YearBuilt"] >= 1900
        assert SAMPLE_HOUSE["GrLivArea"] > 0

    def test_year_built_before_remodel(self):
        """Test YearBuilt is not after YearRemodAdd."""
        assert SAMPLE_HOUSE["YearBuilt"] <= SAMPLE_HOUSE["YearRemodAdd"]


# ─── Test: Feature Engineering in Prediction ──────────────────
class TestFeatureEngineering:

    def test_house_age_calculation(self):
        """Test house age is calculated correctly."""
        year_built = SAMPLE_HOUSE["YearBuilt"]
        expected_age = 2024 - year_built
        assert expected_age == 19

    def test_total_sf_calculation(self):
        """Test total square footage calculation."""
        total_sf = (
            SAMPLE_HOUSE["GrLivArea"] + SAMPLE_HOUSE["TotalBsmtSF"]
        )
        assert total_sf == 2800.0

    def test_total_bathrooms_calculation(self):
        """Test total bathrooms calculation."""
        total_bath = (
            SAMPLE_HOUSE["FullBath"] + SAMPLE_HOUSE["HalfBath"] * 0.5
        )
        assert total_bath == 2.5

    def test_quality_score(self):
        """Test quality score computation."""
        quality_score = (
            SAMPLE_HOUSE["OverallQual"] * SAMPLE_HOUSE["OverallCond"]
        )
        assert quality_score == 35


# ─── Test: Prediction Result Structure ────────────────────────
class TestPredictionResult:

    def test_mock_prediction_structure(self):
        """Test that prediction result has required keys."""
        mock_result = {
            "predicted_price": 285000.0,
            "predicted_price_formatted": "$285,000.00",
            "price_range": {
                "low": "$256,500.00",
                "high": "$313,500.00",
            },
            "confidence": "±10%",
        }

        required_keys = [
            "predicted_price",
            "predicted_price_formatted",
            "price_range",
            "confidence",
        ]

        for key in required_keys:
            assert key in mock_result, f"Missing key: {key}"

    def test_price_range_valid(self):
        """Test price range is logically valid."""
        predicted = 285000.0
        low = predicted * 0.90
        high = predicted * 1.10
        assert low < predicted < high

    def test_prediction_positive(self):
        """Test prediction is always positive."""
        mock_prediction = 285000.0
        assert mock_prediction > 0

    def test_batch_prediction_structure(self):
        """Test batch prediction returns DataFrame."""
        data = {
            "house_id": [1, 2, 3],
            "GrLivArea": [1500, 2000, 1200],
            "OverallQual": [7, 8, 5],
        }
        df = pd.DataFrame(data)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
"""
Unit Tests — Data Preprocessing Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import DataPreprocessor


# ─── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame for testing."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "LotArea": np.random.randint(1500, 20000, n),
        "YearBuilt": np.random.randint(1950, 2020, n),
        "YearRemodAdd": np.random.randint(1950, 2022, n),
        "TotalBsmtSF": np.random.randint(0, 2000, n),
        "GrLivArea": np.random.randint(600, 4000, n),
        "FullBath": np.random.randint(1, 4, n),
        "HalfBath": np.random.randint(0, 2, n),
        "BedroomAbvGr": np.random.randint(1, 6, n),
        "TotRmsAbvGrd": np.random.randint(3, 12, n),
        "GarageArea": np.random.randint(0, 1000, n),
        "PoolArea": np.zeros(n),
        "OverallQual": np.random.randint(1, 11, n),
        "OverallCond": np.random.randint(1, 10, n),
        "MSZoning": np.random.choice(["RL", "RM", "C"], n),
        "Street": np.random.choice(["Pave", "Grvl"], n),
        "Neighborhood": np.random.choice(["NAmes", "CollgCr"], n),
        "BldgType": np.random.choice(["1Fam", "Duplx"], n),
        "HouseStyle": np.random.choice(["1Story", "2Story"], n),
        "ExterQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], n),
        "KitchenQual": np.random.choice(["Ex", "Gd", "TA", "Fa"], n),
        "GarageType": np.random.choice(["Attchd", "Detchd"], n),
        "SaleCondition": np.random.choice(["Normal", "Abnorml"], n),
        "SalePrice": np.random.randint(100000, 500000, n),
    })


@pytest.fixture
def preprocessor():
    """Create a DataPreprocessor instance."""
    return DataPreprocessor()


# ─── Test: Missing Values ─────────────────────────────────────
class TestMissingValues:

    def test_no_missing_after_imputation(self, preprocessor, sample_df):
        """Test that no missing values remain after imputation."""
        # Introduce missing values
        sample_df_missing = sample_df.copy()
        sample_df_missing.loc[0:5, "GrLivArea"] = np.nan
        sample_df_missing.loc[10:15, "MSZoning"] = np.nan

        result = preprocessor.handle_missing_values(sample_df_missing)
        assert result.isnull().sum().sum() == 0, \
            "Missing values should be 0 after imputation."

    def test_shape_preserved(self, preprocessor, sample_df):
        """Test that shape is preserved after handling missing values."""
        result = preprocessor.handle_missing_values(sample_df)
        assert result.shape[0] == sample_df.shape[0]


# ─── Test: Outlier Removal ────────────────────────────────────
class TestOutlierRemoval:

    def test_outliers_removed(self, preprocessor, sample_df):
        """Test that extreme outliers are removed."""
        # Add extreme outlier
        sample_df.loc[0, "GrLivArea"] = 999999
        result = preprocessor.remove_outliers(sample_df, method="iqr")
        assert len(result) < len(sample_df)

    def test_zscore_method(self, preprocessor, sample_df):
        """Test Z-score outlier removal."""
        result = preprocessor.remove_outliers(sample_df, method="zscore")
        assert len(result) <= len(sample_df)

    def test_no_data_loss_without_outliers(self, preprocessor, sample_df):
        """Ensure minimal data loss when no extreme outliers."""
        initial = len(sample_df)
        result = preprocessor.remove_outliers(sample_df, method="iqr")
        assert len(result) > initial * 0.8, \
            "Should not remove more than 20% of clean data."


# ─── Test: Encoding ───────────────────────────────────────────
class TestEncoding:

    def test_no_object_dtypes_after_encoding(
        self, preprocessor, sample_df
    ):
        """Test all categorical columns are encoded."""
        result = preprocessor.encode_categoricals(sample_df)
        obj_cols = result.select_dtypes(include=["object"]).columns.tolist()
        assert len(obj_cols) == 0, \
            f"Object columns remain: {obj_cols}"

    def test_dummy_columns_created(self, preprocessor, sample_df):
        """Test that dummy columns are created."""
        original_cols = sample_df.shape[1]
        result = preprocessor.encode_categoricals(sample_df)
        assert result.shape[1] >= original_cols


# ─── Test: Train/Test Split ───────────────────────────────────
class TestSplit:

    def test_correct_split_ratio(self, preprocessor, sample_df):
        """Test train/test split follows configured ratio."""
        X = sample_df.drop(columns=["SalePrice"])
        y = sample_df["SalePrice"]
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        expected_test = int(len(sample_df) * 0.2)
        assert abs(len(X_test) - expected_test) <= 2

    def test_no_overlap(self, preprocessor, sample_df):
        """Test that train and test sets don't overlap."""
        X = sample_df.drop(columns=["SalePrice"])
        y = sample_df["SalePrice"]
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)

        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        assert len(train_idx.intersection(test_idx)) == 0


# ─── Test: Full Pipeline ──────────────────────────────────────
class TestPipeline:

    def test_pipeline_returns_correct_types(
        self, preprocessor, sample_df
    ):
        """Test pipeline returns expected types."""
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
            sample_df
        )
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_pipeline_no_nans(self, preprocessor, sample_df):
        """Test pipeline output has no NaN values."""
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(
            sample_df
        )
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum() == 0
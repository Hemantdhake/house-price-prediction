"""
Data Preprocessing Module
==========================
Handles all data preprocessing tasks including:
- Missing value imputation
- Outlier detection and treatment
- Feature scaling
- Train/test splitting
"""

import logging
import os
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

# ─── Logger Setup ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Complete data preprocessing pipeline.

    Attributes:
        config (dict): Configuration dictionary.
        numerical_features (list): List of numerical feature names.
        categorical_features (list): List of categorical feature names.
        target_column (str): Name of the target variable.
        scaler: Fitted scaler object.
        num_imputer: Numerical feature imputer.
        cat_imputer: Categorical feature imputer.
    """

    def __init__(self, config_path: str = None):
        """Initialize DataPreprocessor."""
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config.yaml")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.numerical_features   = self.config["data"]["numerical_features"]
        self.categorical_features = self.config["data"]["categorical_features"]
        self.target_column        = self.config["data"]["target_column"]
        self.test_size            = self.config["data"]["test_size"]
        self.random_state         = self.config["data"]["random_state"]

        self.num_imputer = SimpleImputer(
            strategy=self.config["data"]["missing_values"]["numerical_strategy"]
        )
        self.cat_imputer = SimpleImputer(
            strategy=self.config["data"]["missing_values"]["categorical_strategy"]
        )
        self.scaler = RobustScaler()

        logger.info("✅ DataPreprocessor initialized.")

    # ─── Handle Missing Values ────────────────────────────────
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataset.

        Args:
            df: Input DataFrame with potential missing values.

        Returns:
            DataFrame with missing values handled.
        """
        logger.info("🔄 Handling missing values...")
        df = df.copy()

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            logger.info(f"  Found missing values:\n{missing}")
        else:
            logger.info("  No missing values found.")
            return df

        # ── Numerical Imputation ───────────────────────────────
        num_cols = [c for c in self.numerical_features if c in df.columns]
        if num_cols:
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])

        # ── Categorical Imputation ─────────────────────────────
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        if cat_cols:
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])

        # ── Special Cases ──────────────────────────────────────
        if "PoolArea" in df.columns:
            df["PoolArea"] = df["PoolArea"].fillna(0)
        if "GarageType" in df.columns:
            df["GarageType"] = df["GarageType"].fillna("None")
        if "GarageArea" in df.columns:
            df["GarageArea"] = df["GarageArea"].fillna(0)
        if "TotalBsmtSF" in df.columns:
            df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0)

        logger.info(
            f"✅ Missing values handled. "
            f"Remaining: {df.isnull().sum().sum()}"
        )
        return df

    # ─── Remove Outliers ──────────────────────────────────────
    def remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Detect and remove outliers from numerical features.

        Uses a conservative threshold (3.0 IQR) to preserve
        legitimate high-value and distressed properties that
        are real-world cases the model should learn from.

        Args:
            df: Input DataFrame.
            method: Outlier detection method ('iqr' or 'zscore').
            threshold: IQR multiplier or Z-score threshold.
                       Default 3.0 (conservative — keeps edge cases).

        Returns:
            DataFrame with outliers removed.
        """
        logger.info(
            f"🔄 Removing outliers using {method.upper()} "
            f"(threshold={threshold})..."
        )
        df = df.copy()
        initial_rows = len(df)

        # Only apply to core size features — not to quality/year/binary cols
        outlier_cols = [
            c for c in ["LotArea", "GrLivArea", "TotalBsmtSF", "GarageArea"]
            if c in df.columns
        ]

        if method == "iqr":
            for col in outlier_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]

        elif method == "zscore":
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[outlier_cols]))
            df = df[(z_scores < threshold).all(axis=1)]

        removed = initial_rows - len(df)
        logger.info(
            f"✅ Removed {removed} outlier rows "
            f"({removed/initial_rows*100:.1f}%). "
            f"Remaining: {len(df):,}"
        )
        return df

    # ─── Encode Categoricals ──────────────────────────────────
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features.

        Ordinal encoding for quality/condition columns,
        one-hot encoding for all remaining categoricals.

        Args:
            df: Input DataFrame with categorical columns.

        Returns:
            DataFrame with encoded categorical features.
        """
        logger.info("🔄 Encoding categorical features...")
        df = df.copy()

        cat_cols = [c for c in self.categorical_features if c in df.columns]

        # ── Ordinal Encoding for quality columns ──────────────
        quality_map  = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        quality_cols = ["ExterQual", "KitchenQual", "HeatingQC"]
        for col in quality_cols:
            if col in df.columns:
                df[col] = df[col].map(quality_map).fillna(3).astype(int)
                if col in cat_cols:
                    cat_cols.remove(col)

        # ── One-Hot Encoding for remaining categoricals ────────
        remaining_cat = [c for c in cat_cols if c in df.columns]
        if remaining_cat:
            df = pd.get_dummies(df, columns=remaining_cat, drop_first=True)

        # ── Ensure bool columns become int (XGBoost-safe) ─────
        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        logger.info(f"✅ Categorical encoding complete. Shape: {df.shape}")
        return df

    # ─── Scale Features ───────────────────────────────────────
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale numerical features using RobustScaler.

        Fits on training data only, transforms both sets.

        Args:
            X_train: Training feature DataFrame.
            X_test:  Testing feature DataFrame.

        Returns:
            Tuple of (scaled_X_train, scaled_X_test).
        """
        logger.info("🔄 Scaling features...")

        num_cols = [c for c in self.numerical_features if c in X_train.columns]

        X_train = X_train.copy()
        X_test  = X_test.copy()

        if num_cols:
            X_train[num_cols] = self.scaler.fit_transform(X_train[num_cols])
            X_test[num_cols]  = self.scaler.transform(X_test[num_cols])

        logger.info("✅ Feature scaling complete.")
        return X_train, X_test

    # ─── Train/Test Split ─────────────────────────────────────
    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        logger.info(
            f"🔄 Splitting data (test_size={self.test_size}, "
            f"random_state={self.random_state})..."
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        logger.info(
            f"✅ Split complete:\n"
            f"   Training  : {X_train.shape[0]:,} samples\n"
            f"   Testing   : {X_test.shape[0]:,} samples"
        )
        return X_train, X_test, y_train, y_test

    # ─── Full Pipeline ────────────────────────────────────────
    def run_pipeline(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full preprocessing pipeline.

        Steps:
            1. Handle missing values
            2. Remove outliers (conservative threshold)
            3. Encode categoricals
            4. Split features / target
            5. Train/test split
            6. Scale features

        Args:
            df: Raw input DataFrame.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        logger.info("=" * 50)
        logger.info("🚀 RUNNING FULL PREPROCESSING PIPELINE")
        logger.info("=" * 50)

        df = self.handle_missing_values(df)
        df = self.remove_outliers(df, method="iqr", threshold=3.0)
        df = self.encode_categoricals(df)

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        X_train, X_test, y_train, y_test = self.split_data(X, y)
        X_train, X_test = self.scale_features(X_train, X_test)

        logger.info("✅ Preprocessing pipeline complete!")
        logger.info(f"   X_train shape: {X_train.shape}")
        logger.info(f"   X_test shape : {X_test.shape}")

        return X_train, X_test, y_train, y_test


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    df     = loader.load_data()

    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.run_pipeline(df)
    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
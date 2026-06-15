"""
Feature Engineering Module
===========================
Creates new meaningful features from existing ones to
improve model performance.
"""

import logging
from typing import List

import numpy as np
import pandas as pd

# ─── Logger Setup ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates engineered features for house price prediction.

    Methods:
        add_age_features        : House age related features.
        add_area_features       : Total area / bathroom features.
        add_quality_features    : Combined quality score features.
        add_neighborhood_features: Target-encoded neighborhood.
        apply_log_transform     : Log1p on skewed columns.
        run_all                 : Runs all steps in sequence.
    """

    def __init__(self):
        """Initialize FeatureEngineer."""
        self._neighborhood_means: dict = {}   # persists fit for transform
        logger.info("✅ FeatureEngineer initialized.")

    # ─── Age Features ─────────────────────────────────────────
    def add_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add age-related features.

        New Features:
            HouseAge         : current_year - YearBuilt
            RemodelAge       : YearRemodAdd - YearBuilt
            IsRemodeled      : 1 if remodelled, else 0
            YearsAfterRemodel: years since last remodel

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with new age features.
        """
        logger.info("🔄 Adding age features...")
        df = df.copy()
        current_year = 2024

        if "YearBuilt" in df.columns:
            df["HouseAge"] = current_year - df["YearBuilt"]

        if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
            df["RemodelAge"]        = df["YearRemodAdd"] - df["YearBuilt"]
            df["IsRemodeled"]       = (df["RemodelAge"] > 0).astype(int)
            df["YearsAfterRemodel"] = current_year - df["YearRemodAdd"]

        logger.info("✅ Age features added.")
        return df

    # ─── Area Features ────────────────────────────────────────
    def add_area_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add area-related combined features.

        New Features:
            TotalSF        : GrLivArea + TotalBsmtSF
            TotalBathrooms : FullBath + 0.5 * HalfBath
            HasPool        : binary pool indicator
            HasGarage      : binary garage indicator
            LotAreaPerRoom : LotArea / (TotRmsAbvGrd + 1)
            AreaPerBath    : GrLivArea / (TotalBathrooms + 1)

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with new area features.
        """
        logger.info("🔄 Adding area features...")
        df = df.copy()

        sf_cols = [c for c in ["GrLivArea", "TotalBsmtSF"] if c in df.columns]
        if sf_cols:
            df["TotalSF"] = df[sf_cols].sum(axis=1)

        bath_parts = []
        if "FullBath" in df.columns:
            bath_parts.append(df["FullBath"])
        if "HalfBath" in df.columns:
            bath_parts.append(df["HalfBath"] * 0.5)
        if bath_parts:
            df["TotalBathrooms"] = sum(bath_parts)

        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"] > 0).astype(int)

        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

        if "LotArea" in df.columns and "TotRmsAbvGrd" in df.columns:
            df["LotAreaPerRoom"] = df["LotArea"] / (df["TotRmsAbvGrd"] + 1)

        if "GrLivArea" in df.columns and "TotalBathrooms" in df.columns:
            df["AreaPerBath"] = df["GrLivArea"] / (df["TotalBathrooms"] + 1)

        logger.info("✅ Area features added.")
        return df

    # ─── Quality Features ─────────────────────────────────────
    def add_quality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add combined quality score features.

        New Features:
            QualityScore  : OverallQual * OverallCond
            QualCondRatio : OverallQual / (OverallCond + 1)
            IsHighQuality : 1 if OverallQual >= 8

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with quality features.
        """
        logger.info("🔄 Adding quality features...")
        df = df.copy()

        if "OverallQual" in df.columns and "OverallCond" in df.columns:
            df["QualityScore"]  = df["OverallQual"] * df["OverallCond"]
            df["QualCondRatio"] = df["OverallQual"] / (df["OverallCond"] + 1)
            df["IsHighQuality"] = (df["OverallQual"] >= 8).astype(int)

        logger.info("✅ Quality features added.")
        return df

    # ─── Neighborhood Features ────────────────────────────────
    def add_neighborhood_features(
        self,
        df: pd.DataFrame,
        y: pd.Series = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Add target-encoded neighborhood mean price feature.

        Fits on training data (fit=True) and uses stored means
        for inference (fit=False) to prevent data leakage.

        New Features:
            NeighborhoodAvgPrice: mean SalePrice per neighborhood

        Args:
            df:  Input DataFrame.
            y:   Target Series (required when fit=True).
            fit: True during training, False during inference.

        Returns:
            DataFrame with neighborhood feature.
        """
        logger.info("🔄 Adding neighborhood features...")
        df = df.copy()

        if "Neighborhood" not in df.columns:
            return df

        if fit and y is not None:
            means = (
                pd.DataFrame({"Neighborhood": df["Neighborhood"], "Price": y})
                .groupby("Neighborhood")["Price"]
                .mean()
                .to_dict()
            )
            self._neighborhood_means = means

        if self._neighborhood_means:
            global_mean = np.mean(list(self._neighborhood_means.values()))
            df["NeighborhoodAvgPrice"] = (
                df["Neighborhood"]
                .map(self._neighborhood_means)
                .fillna(global_mean)
            )

        logger.info("✅ Neighborhood features added.")
        return df

    # ─── Log Transforms ───────────────────────────────────────
    def apply_log_transform(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Apply log1p transformation to skewed numeric features.

        Skips the target column and any column with negative values.
        Creates new <col>_log columns alongside the originals.

        Args:
            df:      Input DataFrame.
            columns: Columns to transform (auto-detected if None).

        Returns:
            DataFrame with log-transformed features added.
        """
        logger.info("🔄 Applying log transformation...")
        df = df.copy()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Never log-transform the target
        num_cols = [c for c in num_cols if c != "SalePrice"]

        if columns is None:
            skewness = df[num_cols].skew()
            columns  = skewness[skewness.abs() > 0.75].index.tolist()
            logger.info(f"  Auto-detected skewed columns ({len(columns)}): {columns}")

        transformed = []
        for col in columns:
            if col in df.columns and df[col].min() >= 0:
                df[f"{col}_log"] = np.log1p(df[col])
                transformed.append(col)

        logger.info(f"✅ Log transformation applied to {len(transformed)} columns.")
        return df

    # ─── Run All Engineering ──────────────────────────────────
    def run_all(
        self,
        df: pd.DataFrame,
        y: pd.Series = None,
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Execute all feature engineering steps in sequence.

        Args:
            df:  Input DataFrame.
            y:   Target Series (for neighborhood target encoding).
            fit: True during training, False during inference.

        Returns:
            DataFrame with all engineered features.
        """
        logger.info("=" * 50)
        logger.info("🚀 RUNNING FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 50)

        original_cols = len(df.columns)

        df = self.add_age_features(df)
        df = self.add_area_features(df)
        df = self.add_quality_features(df)
        df = self.add_neighborhood_features(df, y=y, fit=fit)
        df = self.apply_log_transform(df)

        new_cols = len(df.columns) - original_cols
        logger.info(
            f"✅ Feature engineering complete! "
            f"Added {new_cols} new features. "
            f"Total: {df.shape[1]} columns."
        )
        return df


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data_loader import DataLoader

    loader = DataLoader()
    df     = loader.load_data()
    X, y   = loader.split_features_target(df)

    engineer   = FeatureEngineer()
    X_engineered = engineer.run_all(X, y)
    print(f"\nOriginal features  : {X.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")
    print(f"\nNew features added:\n{X_engineered.columns.tolist()}")
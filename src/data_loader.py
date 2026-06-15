"""
Data Loader Module
==================
Handles all data loading and initial validation
for the house price prediction project.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ─── Logger Setup ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading, validation, and initial profiling.

    Attributes:
        config_path (str): Path to the configuration YAML file.
        config (dict): Loaded configuration dictionary.
        raw_data_path (str): Path to raw CSV data.
        target_column (str): Name of the target variable.
    """

    def __init__(self, config_path: str = None):
        """
        Initialize DataLoader with config.

        Args:
            config_path: Path to YAML configuration file.
                         Defaults to <project_root>/config/config.yaml.
        """
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config.yaml")

        self.config_path = config_path
        self.config = self._load_config()

        # ── Resolve raw_data_path to absolute ─────────────────
        raw_path = self.config["data"]["raw_data_path"]
        if not os.path.isabs(raw_path):
            base_dir = os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)
            ))
            raw_path = os.path.join(base_dir, raw_path)
        self.raw_data_path = os.path.abspath(raw_path)

        self.target_column = self.config["data"]["target_column"]
        logger.info("✅ DataLoader initialized successfully.")

    # ─── Config Loader ────────────────────────────────────────
    def _load_config(self) -> dict:
        """Load YAML configuration file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"✅ Config loaded from: {self.config_path}")
        return config

    # ─── Generate Sample Data ─────────────────────────────────
    def generate_sample_data(
        self, n_samples: int = 1460, save: bool = True
    ) -> pd.DataFrame:
        """
        Generate realistic housing dataset for demonstration.

        Uses proper skewed distributions and realistic correlations
        instead of uniform random to avoid synthetic-data bias.

        Args:
            n_samples: Number of samples to generate.
            save: Whether to save the generated data to CSV.

        Returns:
            Generated DataFrame.
        """
        np.random.seed(42)
        logger.info(f"🔄 Generating {n_samples} realistic samples...")

        # ── Quality first — drives most other features ─────────
        qual_probs = [.01,.01,.03,.07,.14,.25,.22,.14,.09,.04]
        overall_qual = np.random.choice(range(1, 11), n_samples, p=qual_probs)
        cond_probs   = [.01,.02,.05,.10,.29,.15,.21,.12,.04,.01]
        overall_cond = np.random.choice(range(1, 11), n_samples, p=cond_probs)

        # ── Year features ──────────────────────────────────────
        year_built = np.random.choice(
            range(1872, 2025), n_samples,
            p=self._year_built_probs()
        )
        no_remodel  = np.random.random(n_samples) < 0.55
        year_remod  = np.where(
            no_remodel,
            year_built,
            np.maximum(
                year_built,
                np.random.choice(
                    list(range(1990, 2011)) + list(range(2015, 2025)),
                    n_samples
                )
            )
        )

        # ── Size features (correlated with quality) ────────────
        qual_boost = (overall_qual - 5) * 80
        gr_liv = np.clip(
            np.random.lognormal(7.22, 0.32, n_samples).astype(int) + qual_boost,
            334, 5642
        )
        no_bsmt = np.random.random(n_samples) < 0.08
        bsmt_sf = np.where(
            no_bsmt, 0,
            np.clip((gr_liv * 0.65 + np.random.normal(0, 150, n_samples)).astype(int), 0, 6110)
        )
        lot_area = np.clip(
            np.random.lognormal(9.18, 0.40, n_samples).astype(int), 1300, 215000
        )

        # ── Garage ────────────────────────────────────────────
        no_garage   = np.random.random(n_samples) < 0.06
        garage_area = np.where(
            no_garage, 0,
            np.clip(
                (200 + overall_qual * 35 + (year_built - 1872) * 0.9).astype(int)
                + np.random.normal(0, 70, n_samples).astype(int),
                100, 1418
            )
        )

        # ── Pool ──────────────────────────────────────────────
        has_pool  = (np.random.random(n_samples) < 0.06) & (overall_qual >= 5)
        pool_area = np.where(
            has_pool,
            np.clip(np.random.lognormal(6.2, 0.4, n_samples).astype(int), 120, 738),
            0
        )

        # ── Rooms ─────────────────────────────────────────────
        size_tier = np.clip((gr_liv - 334) / (5642 - 334), 0, 1)
        full_bath = np.clip(
            (size_tier * 3 + np.random.normal(0.5, 0.5, n_samples)).astype(int), 0, 4
        )
        half_bath = np.random.choice([0, 1, 2], n_samples, p=[.60, .36, .04])
        bedroom   = np.clip(
            (size_tier * 4 + np.random.normal(1.5, 0.8, n_samples)).astype(int), 0, 6
        )
        tot_rms   = np.clip(bedroom + full_bath + np.random.randint(2, 5, n_samples), 2, 14)

        # ── Categoricals ──────────────────────────────────────
        def pick(choices, probs):
            p = np.array(probs, dtype=float); p /= p.sum()
            return np.random.choice(choices, n_samples, p=p)

        ms_zoning = pick(
            ["RL", "RM", "FV", "RH", "C (all)", "I"],
            [.79, .11, .04, .03, .02, .01]
        )
        street       = pick(["Pave", "Grvl"], [.996, .004])
        lot_shape    = pick(["Reg", "IR1", "IR2", "IR3"], [.64, .24, .08, .04])
        land_contour = pick(["Lvl", "Bnk", "HLS", "Low"], [.90, .04, .04, .02])
        utilities    = pick(["AllPub", "NoSewr"], [.999, .001])
        neighborhood = pick(
            ["NAmes","CollgCr","OldTown","Edwards","Somerst","NridgHt",
             "Gilbert","Sawyer","NWAmes","SawyerW","Mitchel","BrkSide",
             "Crawfor","IDOTRR","Timber","NoRidge","StoneBr","SWISU",
             "ClearCr","MeadowV"],
            [.15,.10,.08,.07,.06,.06,.05,.05,.05,.04,.04,.03,.03,
             .03,.03,.03,.03,.02,.02,.02]
        )
        bldg_type    = pick(["1Fam","TwnhsE","Duplex","Twnhs","2fmCon"], [.83,.07,.04,.04,.02])
        house_style  = pick(["1Story","2Story","1.5Fin","SLvl","SFoyer"], [.49,.30,.11,.05,.05])
        roof_style   = pick(["Gable","Hip","Gambrel","Flat"], [.78,.19,.02,.01])
        exter_qual   = pick(["TA","Gd","Ex","Fa"], [.62,.28,.07,.03])
        heating_qc   = pick(["Ex","TA","Gd","Fa","Po"], [.51,.24,.19,.05,.01])
        kitchen_qual = pick(["TA","Gd","Ex","Fa"], [.50,.33,.11,.06])
        garage_type  = np.where(
            garage_area == 0, "NA",
            pick(["Attch","Detchd","BuiltIn","CarPort","Basment"], [.62,.28,.07,.02,.01])
        )
        sale_condition = pick(
            ["Normal","Partial","Abnorml","Family","Alloca"],
            [.82,.09,.07,.01,.01]
        )

        # ── SalePrice with realistic formula ──────────────────
        neigh_mult = {
            "NoRidge":1.35,"NridgHt":1.35,"StoneBr":1.35,
            "Timber":1.20,"Somerst":1.20,"CollgCr":1.10,
            "Crawfor":1.10,"ClearCr":1.10,
            "NAmes":1.00,"Gilbert":1.00,"NWAmes":1.00,
            "SawyerW":1.00,"Mitchel":0.95,"Sawyer":0.95,
            "OldTown":0.88,"BrkSide":0.88,"IDOTRR":0.88,
            "Edwards":0.88,"MeadowV":0.82,"SWISU":0.82,
        }
        exter_mult = {"Ex":1.15,"Gd":1.06,"TA":1.00,"Fa":0.90}
        kitch_mult = {"Ex":1.10,"Gd":1.05,"TA":1.00,"Fa":0.91}
        bldg_mult  = {"1Fam":1.0,"TwnhsE":0.95,"Duplex":0.90,"Twnhs":0.90,"2fmCon":0.85}

        nm = np.array([neigh_mult.get(n, 1.0) for n in neighborhood])
        em = np.array([exter_mult[e]           for e in exter_qual])
        km = np.array([kitch_mult[k]           for k in kitchen_qual])
        bm = np.array([bldg_mult[b]            for b in bldg_type])

        qual_factor = np.exp(overall_qual * 0.28)
        base = (
            40_000
            + qual_factor * 8_500
            + overall_cond * 2_000
            + gr_liv * 62
            + bsmt_sf * 18
            + garage_area * 28
            + lot_area * 1.2
            + (year_built - 1872) * 280
            + (year_remod - year_built) * 150
            + full_bath * 6_000
            + half_bath * 3_000
            + np.where(pool_area > 0, 5_000 + pool_area * 40, 0)
        )
        price = base * nm * em * km * bm
        price += np.random.normal(0, price * 0.07)

        abnorm = sale_condition == "Abnorml"
        family = sale_condition == "Family"
        price[abnorm] *= np.random.uniform(0.55, 0.75, abnorm.sum())
        price[family] *= np.random.uniform(0.60, 0.85, family.sum())

        sale_price = np.clip(price, 34_900, 755_000).astype(int)
        sale_price = ((sale_price + 50) // 100) * 100

        data = {
            "LotArea":       lot_area,
            "YearBuilt":     year_built,
            "YearRemodAdd":  year_remod,
            "TotalBsmtSF":   bsmt_sf,
            "GrLivArea":     gr_liv,
            "FullBath":      full_bath,
            "HalfBath":      half_bath,
            "BedroomAbvGr":  bedroom,
            "TotRmsAbvGrd":  tot_rms,
            "GarageArea":    garage_area,
            "PoolArea":      pool_area,
            "OverallQual":   overall_qual,
            "OverallCond":   overall_cond,
            "MSZoning":      ms_zoning,
            "Street":        street,
            "LotShape":      lot_shape,
            "LandContour":   land_contour,
            "Utilities":     utilities,
            "Neighborhood":  neighborhood,
            "BldgType":      bldg_type,
            "HouseStyle":    house_style,
            "RoofStyle":     roof_style,
            "ExterQual":     exter_qual,
            "HeatingQC":     heating_qc,
            "KitchenQual":   kitchen_qual,
            "GarageType":    garage_type,
            "SaleCondition": sale_condition,
            "SalePrice":     sale_price,
        }
        df = pd.DataFrame(data)

        if save:
            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"✅ Data saved to: {self.raw_data_path}")

        return df

    def _year_built_probs(self):
        """Build a probability array for years 1872–2024."""
        years = list(range(1872, 2025))
        n = len(years)
        probs = np.zeros(n)
        for i, yr in enumerate(years):
            if yr < 1920:
                probs[i] = 0.05 / (1920 - 1872)
            elif yr < 1950:
                probs[i] = 0.15 / (1950 - 1920)
            elif yr < 1980:
                probs[i] = 0.35 / (1980 - 1950)
            elif yr < 2000:
                probs[i] = 0.30 / (2000 - 1980)
            else:
                probs[i] = 0.15 / (2025 - 2000)
        return probs / probs.sum()

    # ─── Load Data ────────────────────────────────────────────
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file. Generates realistic sample data
        if the file does not exist.

        Returns:
            Loaded DataFrame.
        """
        if not os.path.exists(self.raw_data_path):
            logger.warning(
                f"⚠️ Data file not found at {self.raw_data_path}. "
                "Generating realistic sample data..."
            )
            return self.generate_sample_data()

        logger.info(f"🔄 Loading data from: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
        return df

    # ─── Data Profiling ───────────────────────────────────────
    def profile_data(self, df: pd.DataFrame) -> dict:
        """
        Generate a comprehensive data profile report.

        Args:
            df: Input DataFrame to profile.

        Returns:
            Dictionary containing profile statistics.
        """
        logger.info("🔄 Generating data profile...")

        numerical_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        profile = {
            "shape":               df.shape,
            "total_rows":          df.shape[0],
            "total_columns":       df.shape[1],
            "numerical_features":  numerical_cols,
            "categorical_features": categorical_cols,
            "missing_values":      df.isnull().sum().to_dict(),
            "missing_percentage":  (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
            "duplicates":          df.duplicated().sum(),
            "memory_usage_mb":     round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "numerical_stats":     df[numerical_cols].describe().to_dict(),
            "target_stats": {
                "mean":     df[self.target_column].mean(),
                "median":   df[self.target_column].median(),
                "std":      df[self.target_column].std(),
                "min":      df[self.target_column].min(),
                "max":      df[self.target_column].max(),
                "skewness": df[self.target_column].skew(),
                "kurtosis": df[self.target_column].kurtosis(),
            },
        }

        print("\n" + "=" * 55)
        print("          📊 DATA PROFILE SUMMARY")
        print("=" * 55)
        print(f"  Total Rows      : {profile['total_rows']:,}")
        print(f"  Total Columns   : {profile['total_columns']}")
        print(f"  Numerical Cols  : {len(numerical_cols)}")
        print(f"  Categorical Cols: {len(categorical_cols)}")
        print(f"  Missing Values  : {sum(df.isnull().sum())}")
        print(f"  Duplicates      : {profile['duplicates']}")
        print(f"  Memory Usage    : {profile['memory_usage_mb']} MB")
        print("-" * 55)
        print(f"  Target Column   : {self.target_column}")
        print(f"  Mean Price      : ${profile['target_stats']['mean']:,.2f}")
        print(f"  Median Price    : ${profile['target_stats']['median']:,.2f}")
        print(f"  Price Range     : ${profile['target_stats']['min']:,} - "
              f"${profile['target_stats']['max']:,}")
        print(f"  Skewness        : {profile['target_stats']['skewness']:.4f}")
        print("=" * 55)

        logger.info("✅ Data profiling complete.")
        return profile

    # ─── Split Features/Target ────────────────────────────────
    def split_features_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features and target.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (X, y) where X is features and y is target.
        """
        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame."
            )
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        logger.info(f"✅ Split complete → X: {X.shape}, y: {y.shape}")
        return X, y


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    loader = DataLoader()
    df     = loader.load_data()
    profile = loader.profile_data(df)
    X, y   = loader.split_features_target(df)
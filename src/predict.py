# """
# Prediction Module
# =================
# Provides prediction interface for the trained model,
# supporting single, batch, and real-time predictions.
# """

# import logging
# import os
# from typing import Dict, List, Union, Optional

# import joblib
# import numpy as np
# import pandas as pd
# import yaml

# # ─── Logger Setup ─────────────────────────────────────────────
# logger = logging.getLogger(__name__)


# class HousePricePredictor:
#     """
#     Provides prediction interface using the trained model.

#     Attributes:
#         config (dict): Configuration dictionary.
#         model: Loaded trained model.
#         scaler: Loaded data scaler.
#         feature_names (list): Expected feature names.
#     """

#     def __init__(self, config_path: str = None):
#         """Initialize predictor and load model."""

#         if config_path is None:
#             base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#             config_path = os.path.join(base_dir, "config", "config.yaml")

#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)

#         self.model_path = self.config["model"]["save_path"]
#         self.scaler_path = self.config["model"]["scaler_path"]
#         self.numerical_features = self.config["data"]["numerical_features"]

#         # Make artifact paths absolute
#         if not os.path.isabs(self.model_path):
#             self.model_path = os.path.join(
#                 os.path.dirname(config_path), "..", self.model_path
#             )
#         if not os.path.isabs(self.scaler_path):
#             self.scaler_path = os.path.join(
#                 os.path.dirname(config_path), "..", self.scaler_path
#             )

#         self.model_path  = os.path.abspath(self.model_path)
#         self.scaler_path = os.path.abspath(self.scaler_path)

#         self.model  = None
#         self.scaler = None

#         self._load_artifacts()
#         logger.info("✅ HousePricePredictor initialized.")

#     # ─── Load Model Artifacts ─────────────────────────────────
#     def _load_artifacts(self):
#         """Load trained model and scaler from disk."""
#         if os.path.exists(self.model_path):
#             self.model = joblib.load(self.model_path)
#             self._fix_xgboost_compat()          # ← version-mismatch fix
#             logger.info(f"✅ Model loaded: {self.model_path}")
#         else:
#             logger.warning(
#                 f"⚠️ Model not found at {self.model_path}. "
#                 "Train the model first."
#             )

#         if os.path.exists(self.scaler_path):
#             self.scaler = joblib.load(self.scaler_path)
#             logger.info(f"✅ Scaler loaded: {self.scaler_path}")

#     # ─── XGBoost Version Compatibility Fix ────────────────────
#     def _fix_xgboost_compat(self):
#         """
#         Fix attribute mismatches between XGBoost versions.

#         XGBoost < 2.0 stored 'gpu_id', 'predictor', 'n_gpus' in the
#         sklearn wrapper params and in the booster config.
#         XGBoost >= 2.0 removed / renamed these to 'device', causing
#         AttributeError: 'XGBModel' object has no attribute 'gpu_id'
#         on the first predict() call.

#         Strategy:
#           1. Remove stale keys from the sklearn-wrapper param dict.
#           2. Strip them from the booster's JSON config.
#           3. Re-serialise the booster through XGBoost's own native
#              format so any remaining legacy state is purged.
#         """
#         if self.model is None:
#             return

#         try:
#             import xgboost as xgb

#             stale_params = [
#                 "gpu_id", "predictor", "gpu_platform_id",
#                 "gpu_device_id", "n_gpus",
#             ]

#             # ── 1. Sklearn-wrapper level ───────────────────────
#             if hasattr(self.model, "get_params"):
#                 current = self.model.get_params()
#                 remove  = {k: None for k in stale_params if k in current}
#                 if remove:
#                     self.model.set_params(**remove)
#                     logger.info(
#                         f"XGBoost compat: removed stale wrapper params "
#                         f"{list(remove.keys())}"
#                     )

#             # ── 2. Booster JSON config level ───────────────────
#             booster = (
#                 self.model.get_booster()
#                 if hasattr(self.model, "get_booster")
#                 else None
#             )
#             if booster is not None:
#                 import json
#                 cfg_dict = json.loads(booster.save_config())
#                 try:
#                     tp      = cfg_dict["learner"]["gradient_booster"][
#                                   "tree_train_param"
#                               ]
#                     cleaned = [k for k in stale_params if tp.pop(k, None) is not None]
#                     if cleaned:
#                         booster.load_config(json.dumps(cfg_dict))
#                         logger.info(
#                             f"XGBoost compat: cleaned booster config params "
#                             f"{cleaned}"
#                         )
#                 except KeyError:
#                     pass  # key path absent — nothing to clean

#             # ── 3. Re-serialise through native XGBoost format ──
#             # Saves + reloads the booster using XGBoost's own
#             # binary format, which strips any legacy attributes
#             # that joblib preserved but the current version rejects.
#             if booster is not None:
#                 import tempfile
#                 with tempfile.NamedTemporaryFile(
#                     suffix=".ubj", delete=False
#                 ) as tmp:
#                     tmp_path = tmp.name
#                 try:
#                     booster.save_model(tmp_path)
#                     booster.load_model(tmp_path)
#                     logger.info(
#                         "XGBoost compat: re-serialised booster via "
#                         "native format — all stale attributes cleared."
#                     )
#                 finally:
#                     if os.path.exists(tmp_path):
#                         os.remove(tmp_path)

#         except Exception as exc:
#             # Never let a compat fix block a prediction
#             logger.warning(f"XGBoost compat fix skipped ({exc})")

#     # ─── Preprocess Input ─────────────────────────────────────
#     def preprocess_input(self, input_data: Dict) -> pd.DataFrame:
#         """
#         Preprocess a single input dictionary for prediction.

#         Args:
#             input_data: Dictionary of feature values.

#         Returns:
#             Preprocessed DataFrame ready for prediction.
#         """
#         df = pd.DataFrame([input_data])

#         # ── Feature Engineering ────────────────────────────────
#         current_year = 2024
#         if "YearBuilt" in df.columns:
#             df["HouseAge"] = current_year - df["YearBuilt"]
#         if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
#             df["RemodelAge"]  = df["YearRemodAdd"] - df["YearBuilt"]
#             df["IsRemodeled"] = (df["RemodelAge"] > 0).astype(int)

#         if "GrLivArea" in df.columns and "TotalBsmtSF" in df.columns:
#             df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

#         if "FullBath" in df.columns:
#             df["TotalBathrooms"] = (
#                 df.get("FullBath", 0) + df.get("HalfBath", 0) * 0.5
#             )

#         if "OverallQual" in df.columns and "OverallCond" in df.columns:
#             df["QualityScore"] = df["OverallQual"] * df["OverallCond"]
#             df["IsHighQuality"] = (df["OverallQual"] >= 8).astype(int)

#         # ── Scale Numerical Features ───────────────────────────
#         num_cols = [c for c in self.numerical_features if c in df.columns]
#         if self.scaler and num_cols:
#             df[num_cols] = self.scaler.transform(df[num_cols])

#         return df

#     # ─── Single Prediction ────────────────────────────────────
#     def predict_single(self, input_data: Dict) -> Dict:
#         """
#         Make a prediction for a single house.

#         Args:
#             input_data: Dictionary with house features.

#         Returns:
#             Dictionary containing prediction and confidence.
#         """
#         if self.model is None:
#             raise RuntimeError("Model not loaded. Train the model first.")

#         df = self.preprocess_input(input_data)

#         # Align with model features if needed
#         if hasattr(self.model, "feature_names_in_"):
#             for col in self.model.feature_names_in_:
#                 if col not in df.columns:
#                     df[col] = 0
#             df = df[self.model.feature_names_in_]

#         prediction = self.model.predict(df)[0]
#         prediction = max(0, prediction)

#         result = {
#             "predicted_price": round(float(prediction), 2),
#             "predicted_price_formatted": f"${prediction:,.2f}",
#             "price_range": {
#                 "low":  f"${prediction * 0.90:,.2f}",
#                 "high": f"${prediction * 1.10:,.2f}",
#             },
#             "confidence": "±10%",
#         }
#         logger.info(f"✅ Prediction: {result['predicted_price_formatted']}")
#         return result

#     # ─── Batch Prediction ─────────────────────────────────────
#     def predict_batch(
#         self,
#         df: pd.DataFrame,
#         output_path: Optional[str] = None,
#     ) -> pd.DataFrame:
#         """
#         Make predictions for a batch of houses.

#         Args:
#             df: DataFrame with house features.
#             output_path: Optional path to save results.

#         Returns:
#             DataFrame with predictions added.
#         """
#         if self.model is None:
#             raise RuntimeError("Model not loaded.")

#         logger.info(f"🔄 Running batch prediction on {len(df)} samples...")

#         df_processed = df.copy()

#         current_year = 2024
#         if "YearBuilt" in df_processed.columns:
#             df_processed["HouseAge"] = current_year - df_processed["YearBuilt"]
#         if "GrLivArea" in df_processed.columns:
#             df_processed["TotalSF"] = (
#                 df_processed.get("GrLivArea", 0)
#                 + df_processed.get("TotalBsmtSF", 0)
#             )

#         num_cols = [c for c in self.numerical_features if c in df_processed.columns]
#         if self.scaler and num_cols:
#             df_processed[num_cols] = self.scaler.transform(df_processed[num_cols])

#         if hasattr(self.model, "feature_names_in_"):
#             for col in self.model.feature_names_in_:
#                 if col not in df_processed.columns:
#                     df_processed[col] = 0
#             df_processed = df_processed[self.model.feature_names_in_]

#         predictions = self.model.predict(df_processed)
#         df["PredictedPrice"] = np.round(predictions, 2)
#         df["PredictedPrice"] = df["PredictedPrice"].clip(lower=0)

#         if output_path:
#             df.to_csv(output_path, index=False)
#             logger.info(f"✅ Predictions saved to: {output_path}")

#         logger.info(
#             f"✅ Batch prediction complete. "
#             f"Mean prediction: ${df['PredictedPrice'].mean():,.2f}"
#         )
#         return df

#     # ─── What-If Analysis ─────────────────────────────────────
#     def what_if_analysis(
#         self,
#         base_input: Dict,
#         feature: str,
#         values: List,
#     ) -> pd.DataFrame:
#         """
#         Perform what-if analysis by varying a single feature.

#         Args:
#             base_input: Base house feature dictionary.
#             feature: Feature to vary.
#             values: List of values to test.

#         Returns:
#             DataFrame showing price predictions for each value.
#         """
#         logger.info(f"🔄 What-if analysis for feature: {feature}")

#         results = []
#         for val in values:
#             test_input = base_input.copy()
#             test_input[feature] = val
#             pred = self.predict_single(test_input)
#             results.append({
#                 feature: val,
#                 "PredictedPrice": pred["predicted_price"],
#             })

#         df = pd.DataFrame(results)
#         print(f"\n📊 What-If Analysis: {feature}")
#         print(df.to_string(index=False))
#         return df


# # ─── Main ─────────────────────────────────────────────────────
# def main():
#     """Example usage of HousePricePredictor."""
#     predictor = HousePricePredictor()

#     sample_house = {
#         "LotArea":      8500,
#         "YearBuilt":    2005,
#         "YearRemodAdd": 2015,
#         "TotalBsmtSF":  1000,
#         "GrLivArea":    1800,
#         "FullBath":        2,
#         "HalfBath":        1,
#         "BedroomAbvGr":    3,
#         "TotRmsAbvGrd":    7,
#         "GarageArea":    400,
#         "PoolArea":        0,
#         "OverallQual":     7,
#         "OverallCond":     5,
#     }

#     result = predictor.predict_single(sample_house)
#     print(f"\n🏠 Predicted House Price: {result['predicted_price_formatted']}")
#     print(f"   Price Range: {result['price_range']['low']} — "
#           f"{result['price_range']['high']}")


# if __name__ == "__main__":
#     main()

"""
Prediction Module
=================
Provides prediction interface for the trained model,
supporting single, batch, and real-time predictions.
"""

import logging
import os
from typing import Dict, List, Union, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

# ─── Logger Setup ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """
    Provides prediction interface using the trained model.

    Attributes:
        config (dict): Configuration dictionary.
        model: Loaded trained model.
        scaler: Loaded data scaler.
        feature_names (list): Expected feature names.
    """

    def __init__(self, config_path: str = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                "Make sure config/config.yaml exists in your project root."
            )
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model"]["save_path"]
        self.scaler_path = self.config["model"]["scaler_path"]
        self.numerical_features = self.config["data"]["numerical_features"]

        self.model = None
        self.scaler = None

        self._load_artifacts()
        logger.info("✅ HousePricePredictor initialized.")

    # ─── Load Model Artifacts ─────────────────────────────────
    def _load_artifacts(self):
        """Load trained model and scaler from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ Model loaded: {self.model_path}")
        else:
            logger.warning(
                f"⚠️ Model not found at {self.model_path}. "
                "Train the model first."
            )

        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"✅ Scaler loaded: {self.scaler_path}")

    # ─── Preprocess Input ─────────────────────────────────────
    def preprocess_input(
        self, input_data: Dict
    ) -> pd.DataFrame:
        """
        Preprocess a single input dictionary for prediction.

        Args:
            input_data: Dictionary of feature values.

        Returns:
            Preprocessed DataFrame ready for prediction.
        """
        df = pd.DataFrame([input_data])

        # ── Feature Engineering ────────────────────────────────
        current_year = 2024
        if "YearBuilt" in df.columns:
            df["HouseAge"] = current_year - df["YearBuilt"]
        if "YearRemodAdd" in df.columns and "YearBuilt" in df.columns:
            df["RemodelAge"] = df["YearRemodAdd"] - df["YearBuilt"]
            df["IsRemodeled"] = (df["RemodelAge"] > 0).astype(int)

        if "GrLivArea" in df.columns and "TotalBsmtSF" in df.columns:
            df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]

        if "FullBath" in df.columns:
            df["TotalBathrooms"] = (
                df.get("FullBath", 0) + df.get("HalfBath", 0) * 0.5
            )

        if "OverallQual" in df.columns and "OverallCond" in df.columns:
            df["QualityScore"] = df["OverallQual"] * df["OverallCond"]
            df["IsHighQuality"] = (df["OverallQual"] >= 8).astype(int)

        # ── Scale Numerical Features ───────────────────────────
        num_cols = [
            c for c in self.numerical_features if c in df.columns
        ]
        if self.scaler and num_cols:
            df[num_cols] = self.scaler.transform(df[num_cols])

        return df

    # ─── Single Prediction ────────────────────────────────────
    def predict_single(self, input_data: Dict) -> Dict:
        """
        Make a prediction for a single house.

        Args:
            input_data: Dictionary with house features.

        Returns:
            Dictionary containing prediction and confidence.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Train the model first."
            )

        df = self.preprocess_input(input_data)

        # Align with model features if needed
        if hasattr(self.model, "feature_names_in_"):
            for col in self.model.feature_names_in_:
                if col not in df.columns:
                    df[col] = 0
            df = df[self.model.feature_names_in_]

        prediction = self.model.predict(df)[0]
        prediction = max(0, prediction)

        result = {
            "predicted_price": round(float(prediction), 2),
            "predicted_price_formatted": f"${prediction:,.2f}",
            "price_range": {
                "low": f"${prediction * 0.90:,.2f}",
                "high": f"${prediction * 1.10:,.2f}",
            },
            "confidence": "±10%",
        }

        logger.info(
            f"✅ Prediction: {result['predicted_price_formatted']}"
        )
        return result

    # ─── Batch Prediction ─────────────────────────────────────
    def predict_batch(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Make predictions for a batch of houses.

        Args:
            df: DataFrame with house features.
            output_path: Optional path to save results.

        Returns:
            DataFrame with predictions added.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        logger.info(f"🔄 Running batch prediction on {len(df)} samples...")

        df_processed = df.copy()

        # Apply feature engineering
        current_year = 2024
        if "YearBuilt" in df_processed.columns:
            df_processed["HouseAge"] = current_year - df_processed["YearBuilt"]
        if "GrLivArea" in df_processed.columns:
            df_processed["TotalSF"] = (
                df_processed.get("GrLivArea", 0)
                + df_processed.get("TotalBsmtSF", 0)
            )

        # Scale
        num_cols = [
            c for c in self.numerical_features if c in df_processed.columns
        ]
        if self.scaler and num_cols:
            df_processed[num_cols] = self.scaler.transform(
                df_processed[num_cols]
            )

        # Align features
        if hasattr(self.model, "feature_names_in_"):
            for col in self.model.feature_names_in_:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            df_processed = df_processed[self.model.feature_names_in_]

        predictions = self.model.predict(df_processed)
        df["PredictedPrice"] = np.round(predictions, 2)
        df["PredictedPrice"] = df["PredictedPrice"].clip(lower=0)

        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"✅ Predictions saved to: {output_path}")

        logger.info(
            f"✅ Batch prediction complete. "
            f"Mean prediction: ${df['PredictedPrice'].mean():,.2f}"
        )
        return df

    # ─── What-If Analysis ─────────────────────────────────────
    def what_if_analysis(
        self,
        base_input: Dict,
        feature: str,
        values: List,
    ) -> pd.DataFrame:
        """
        Perform what-if analysis by varying a single feature.

        Args:
            base_input: Base house feature dictionary.
            feature: Feature to vary.
            values: List of values to test.

        Returns:
            DataFrame showing price predictions for each value.
        """
        logger.info(f"🔄 What-if analysis for feature: {feature}")

        results = []
        for val in values:
            test_input = base_input.copy()
            test_input[feature] = val
            pred = self.predict_single(test_input)
            results.append({
                feature: val,
                "PredictedPrice": pred["predicted_price"],
            })

        df = pd.DataFrame(results)
        print(f"\n📊 What-If Analysis: {feature}")
        print(df.to_string(index=False))
        return df


# ─── Main ─────────────────────────────────────────────────────
def main():
    """Example usage of HousePricePredictor."""
    predictor = HousePricePredictor()

    # Sample house
    sample_house = {
        "LotArea": 8500,
        "YearBuilt": 2005,
        "YearRemodAdd": 2015,
        "TotalBsmtSF": 1000,
        "GrLivArea": 1800,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "TotRmsAbvGrd": 7,
        "GarageArea": 400,
        "PoolArea": 0,
        "OverallQual": 7,
        "OverallCond": 5,
    }

    result = predictor.predict_single(sample_house)
    print(f"\n🏠 Predicted House Price: {result['predicted_price_formatted']}")
    print(f"   Price Range: {result['price_range']['low']} - "
          f"{result['price_range']['high']}")


if __name__ == "__main__":
    main()
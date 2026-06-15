"""
Model Training Module
=====================
Trains, tunes, and selects the best regression model
for house price prediction.
"""

import json
import logging
import os
import tempfile
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Logger Setup ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Manages model training, hyperparameter tuning,
    cross-validation, and model persistence.

    Attributes:
        config (dict): Configuration dictionary.
        models (dict): Dictionary of model instances.
        results (dict): Training results for each model.
        best_model: The best performing model.
        best_model_name (str): Name of the best model.
    """

    def __init__(self, config_path: str = None):
        """Initialize ModelTrainer."""
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "config.yaml")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_config  = self.config["model"]
        self.cv_config     = self.config["model"]["cv"]
        self.random_state  = self.config["data"]["random_state"]

        # Resolve model/scaler save paths to absolute
        self._model_save_path  = self._abs(self.model_config["save_path"],  config_path)
        self._scaler_save_path = self._abs(self.model_config["scaler_path"], config_path)

        self.models         = {}
        self.results        = {}
        self.best_model     = None
        self.best_model_name = None

        self._initialize_models()
        logger.info("✅ ModelTrainer initialized.")

    # ─── Path Helper ──────────────────────────────────────────
    @staticmethod
    def _abs(path: str, config_path: str) -> str:
        """Resolve a potentially relative path against the project root."""
        if os.path.isabs(path):
            return path
        project_root = os.path.dirname(os.path.dirname(config_path))
        return os.path.abspath(os.path.join(project_root, path))

    # ─── Initialize Models ────────────────────────────────────
    def _initialize_models(self):
        """Initialize all regression models."""
        self.models = {
            "linear_regression": LinearRegression(),
            "ridge":             Ridge(random_state=self.random_state),
            "lasso":             Lasso(random_state=self.random_state),
            "decision_tree":     DecisionTreeRegressor(random_state=self.random_state),
            "random_forest":     RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state,
            ),
            # XGBoost 2.x compatible — no gpu_id / predictor params
            "xgboost": xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                device="cpu",           # explicit — avoids legacy gpu_id
                tree_method="hist",     # fast CPU tree method
            ),
        }
        logger.info(
            f"✅ Initialized {len(self.models)} models: "
            f"{list(self.models.keys())}"
        )

    # ─── Baseline Training ────────────────────────────────────
    def train_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
    ) -> Dict[str, Dict]:
        """
        Train all models with default parameters.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_test:  Testing features.
            y_test:  Testing target.

        Returns:
            Dictionary with model results.
        """
        logger.info("=" * 55)
        logger.info("🚀 BASELINE MODEL TRAINING")
        logger.info("=" * 55)

        print("\n" + "─" * 55)
        print(f"{'Model':<25} {'R²':>8} {'RMSE':>12} {'Time':>8}")
        print("─" * 55)

        for name, model in self.models.items():
            start = time.time()
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_pred_test  = model.predict(X_test)

            train_r2  = r2_score(y_train, y_pred_train)
            test_r2   = r2_score(y_test,  y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            elapsed   = time.time() - start

            self.results[name] = {
                "model":      model,
                "train_r2":   round(train_r2,  4),
                "test_r2":    round(test_r2,   4),
                "test_rmse":  round(test_rmse, 2),
                "train_time": round(elapsed,   2),
                "y_pred":     y_pred_test,
            }
            print(
                f"{name:<25} {test_r2:>8.4f} "
                f"${test_rmse:>11,.0f} {elapsed:>6.1f}s"
            )

        print("─" * 55)
        logger.info("✅ Baseline training complete.")
        return self.results

    # ─── Cross Validation ─────────────────────────────────────
    def cross_validate_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, Dict]:
        """
        Run K-Fold cross-validation for all models.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            Cross-validation results.
        """
        logger.info("🔄 Running cross-validation...")

        cv_results = {}
        kfold = KFold(
            n_splits=self.cv_config["n_splits"],
            shuffle=True,
            random_state=self.random_state,
        )

        print("\n" + "─" * 60)
        print(f"{'Model':<25} {'CV Mean R²':>12} {'CV Std':>10}")
        print("─" * 60)

        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=kfold, scoring="r2", n_jobs=-1)
            cv_results[name] = {
                "cv_scores": scores,
                "cv_mean":   round(scores.mean(), 4),
                "cv_std":    round(scores.std(),  4),
            }
            print(f"{name:<25} {scores.mean():>12.4f} ±{scores.std():>8.4f}")

        print("─" * 60)
        logger.info("✅ Cross-validation complete.")
        return cv_results

    # ─── Hyperparameter Tuning ────────────────────────────────
    def tune_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str = "xgboost",
    ) -> Any:
        """
        Perform GridSearchCV hyperparameter tuning.

        XGBoost param grid uses 2.x-compatible keys only
        (no gpu_id / predictor / n_gpus).

        Args:
            X_train:    Training features.
            y_train:    Training target.
            model_name: Name of model to tune.

        Returns:
            Tuned model (best estimator from GridSearchCV).
        """
        logger.info(f"🔄 Tuning hyperparameters for: {model_name}")

        param_grids = {
            "xgboost": {
                "n_estimators":    [100, 200, 300],
                "learning_rate":   [0.05, 0.1, 0.2],
                "max_depth":       [3, 5, 7],
                "subsample":       [0.8, 1.0],
                "colsample_bytree":[0.8, 1.0],
                "min_child_weight":[1, 3],
                "reg_alpha":       [0, 0.1],
                "reg_lambda":      [1, 1.5],
            },
            "random_forest": {
                "n_estimators":    [100, 200],
                "max_depth":       [None, 10, 20],
                "min_samples_split":[2, 5],
                "min_samples_leaf": [1, 2],
            },
            "gradient_boosting": {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5],
            },
        }

        if model_name not in param_grids:
            logger.warning(
                f"No parameter grid for '{model_name}'. "
                "Returning default model."
            )
            return self.models[model_name]

        grid_search = GridSearchCV(
            estimator=self.models[model_name],
            param_grid=param_grids[model_name],
            cv=3,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        logger.info(
            f"✅ Best params : {grid_search.best_params_}\n"
            f"   Best CV R² : {grid_search.best_score_:.4f}"
        )
        return grid_search.best_estimator_

    # ─── Select Best Model ────────────────────────────────────
    def select_best_model(
        self,
        results: Dict[str, Dict],
        metric: str = "test_r2",
    ) -> Tuple[str, Any]:
        """
        Select the best performing model based on a metric.

        Args:
            results: Dictionary of model results.
            metric:  Metric key to rank by.

        Returns:
            Tuple of (best_model_name, best_model).
        """
        best_name  = max(results, key=lambda x: results[x][metric])
        best_model = results[best_name]["model"]

        self.best_model_name = best_name
        self.best_model      = best_model

        logger.info(
            f"✅ Best Model: {best_name} "
            f"(R² = {results[best_name][metric]:.4f})"
        )
        return best_name, best_model

    # ─── Feature Importance ───────────────────────────────────
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Extract and display feature importance from tree models.

        Args:
            model:         Trained model with feature_importances_.
            feature_names: List of feature names.
            top_n:         Number of top features to return.

        Returns:
            DataFrame with feature importances sorted descending.
        """
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model has no feature_importances_ attribute.")
            return pd.DataFrame()

        importance_df = (
            pd.DataFrame({
                "Feature":    feature_names,
                "Importance": model.feature_importances_,
            })
            .sort_values("Importance", ascending=False)
            .head(top_n)
        )

        print("\n" + "─" * 45)
        print(f"  📊 TOP {top_n} FEATURE IMPORTANCES")
        print("─" * 45)
        for _, row in importance_df.iterrows():
            bar = "█" * int(row["Importance"] * 200)
            print(f"  {row['Feature']:<20} {bar:<20} {row['Importance']:.4f}")
        print("─" * 45)

        return importance_df

    # ─── Save Model ───────────────────────────────────────────
    def save_model(
        self,
        model:        Any,
        filepath:     str = None,
        scaler:       Any = None,
        scaler_path:  str = None,
    ):
        """
        Save trained model (and optionally scaler) to disk.

        For XGBoost models the booster is also re-serialised
        through XGBoost's native format so the saved pickle is
        free of legacy attributes (gpu_id etc.) from the start.

        Args:
            model:       Trained model to save.
            filepath:    Path to save the model (.pkl).
            scaler:      Scaler object to save.
            scaler_path: Path to save the scaler (.pkl).
        """
        filepath    = filepath    or self._model_save_path
        scaler_path = scaler_path or self._scaler_save_path

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # ── XGBoost: purge legacy booster state before saving ──
        if hasattr(model, "get_booster"):
            self._purge_xgb_legacy(model)

        joblib.dump(model, filepath)
        logger.info(f"✅ Model saved to: {filepath}")

        if scaler is not None:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"✅ Scaler saved to: {scaler_path}")

    # ─── XGBoost Legacy Purge ─────────────────────────────────
    @staticmethod
    def _purge_xgb_legacy(model: Any):
        """
        Remove XGBoost 1.x legacy attributes before saving.

        Strips gpu_id, predictor, n_gpus from both the sklearn
        wrapper and the booster's internal JSON config, then
        re-serialises the booster via XGBoost's native binary
        format so the pickle is clean for XGBoost 2.x.

        Args:
            model: XGBoost sklearn-API model.
        """
        stale = ["gpu_id", "predictor", "gpu_platform_id",
                 "gpu_device_id", "n_gpus", "single_precision_histogram"]

        # 1. Sklearn wrapper __dict__
        for key in stale:
            if hasattr(model, key):
                try:
                    delattr(model, key)
                except AttributeError:
                    pass

        # 2. set_params to None
        try:
            current = model.get_params()
            remove  = {k: None for k in stale if k in current}
            if remove:
                model.set_params(**remove)
        except Exception:
            pass

        # 3. Booster JSON config
        try:
            booster  = model.get_booster()
            cfg_dict = json.loads(booster.save_config())
            sections = [
                ["learner", "gradient_booster", "tree_train_param"],
                ["learner", "learner_train_param"],
                ["learner", "generic_param"],
            ]
            changed = False
            for path in sections:
                node = cfg_dict
                try:
                    for key in path:
                        node = node[key]
                    for param in stale:
                        if param in node:
                            del node[param]
                            changed = True
                except KeyError:
                    pass
            if changed:
                booster.load_config(json.dumps(cfg_dict))
        except Exception:
            pass

        # 4. Native re-serialise (definitive purge)
        try:
            booster = model.get_booster()
            with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as tmp:
                tmp_path = tmp.name
            booster.save_model(tmp_path)
            booster.load_model(tmp_path)
            os.remove(tmp_path)
        except Exception:
            pass

    # ─── Load Model ───────────────────────────────────────────
    def load_model(self, filepath: str = None) -> Any:
        """
        Load a trained model from disk.

        Args:
            filepath: Path to the saved model.

        Returns:
            Loaded model object.
        """
        filepath = filepath or self._model_save_path

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        logger.info(f"✅ Model loaded from: {filepath}")
        return model

    # ─── Full Training Pipeline ───────────────────────────────
    def main(self):
        """Run the complete model training pipeline."""
        from src.data_loader        import DataLoader
        from src.preprocessing      import DataPreprocessor
        from src.feature_engineering import FeatureEngineer

        logger.info("=" * 55)
        logger.info("🏠 HOUSE PRICE PREDICTION — TRAINING PIPELINE")
        logger.info("=" * 55)

        # Step 1: Load Data
        loader = DataLoader()
        df     = loader.load_data()

        # Step 2: Feature Engineering
        X, y    = loader.split_features_target(df)
        engineer = FeatureEngineer()
        X        = engineer.run_all(X, y, fit=True)

        # Recombine for preprocessing
        df_engineered = X.copy()
        df_engineered[self.config["data"]["target_column"]] = y.values

        # Step 3: Preprocessing
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.run_pipeline(df_engineered)

        # Step 4: Baseline Training
        results = self.train_baseline_models(X_train, y_train, X_test, y_test)

        # Step 5: Cross Validation
        self.cross_validate_models(X_train, y_train)

        # Step 6: Select Best Model
        best_name, _ = self.select_best_model(results)

        # Step 7: Tune Best Model
        tuned_model = self.tune_best_model(X_train, y_train, best_name)

        # Step 8: Re-evaluate tuned model
        y_pred    = tuned_model.predict(X_test)
        tuned_r2  = r2_score(y_test, y_pred)
        tuned_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logger.info(
            f"✅ Tuned model — R²: {tuned_r2:.4f}  "
            f"RMSE: ${tuned_rmse:,.0f}"
        )

        # Step 9: Feature Importance
        self.get_feature_importance(tuned_model, X_train.columns.tolist())

        # Step 10: Save Model (with legacy purge baked in)
        self.save_model(tuned_model, scaler=preprocessor.scaler)

        logger.info("🎉 Training pipeline complete!")
        return tuned_model, preprocessor


# ─── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.main()
"""
Unit Tests — Model Training Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.model import ModelTrainer


# ─── Fixtures ─────────────────────────────────────────────────
@pytest.fixture
def trainer():
    return ModelTrainer()


@pytest.fixture
def sample_xy():
    """Simple numeric dataset for model testing."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feature_1": np.random.rand(n) * 1000,
        "feature_2": np.random.randint(1, 10, n),
        "feature_3": np.random.rand(n) * 500,
        "feature_4": np.random.randint(0, 5, n),
        "feature_5": np.random.rand(n) * 2000,
    })
    y = (
        X["feature_1"] * 100
        + X["feature_2"] * 5000
        + X["feature_3"] * 50
        + np.random.normal(0, 5000, n)
    )
    return X, pd.Series(y, name="SalePrice")


@pytest.fixture
def split_data(sample_xy):
    """Return train/test split."""
    X, y = sample_xy
    split_idx = int(len(X) * 0.8)
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
    )


# ─── Test: Model Initialization ───────────────────────────────
class TestModelInitialization:

    def test_models_initialized(self, trainer):
        """Test all models are initialized."""
        assert len(trainer.models) > 0

    def test_model_keys(self, trainer):
        """Test expected model keys exist."""
        expected = [
            "linear_regression",
            "random_forest",
            "gradient_boosting",
            "xgboost",
        ]
        for key in expected:
            assert key in trainer.models, f"Missing model: {key}"


# ─── Test: Baseline Training ──────────────────────────────────
class TestBaselineTraining:

    def test_all_models_trained(self, trainer, split_data):
        """Test all models complete training."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        assert len(results) == len(trainer.models)

    def test_results_have_metrics(self, trainer, split_data):
        """Test result dict has required metrics."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        required = ["test_r2", "test_rmse", "train_r2"]
        for name, result in results.items():
            for metric in required:
                assert metric in result, \
                    f"Missing metric '{metric}' in {name}"

    def test_r2_range(self, trainer, split_data):
        """Test R² scores are in valid range."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        for name, result in results.items():
            assert -1 <= result["test_r2"] <= 1, \
                f"R² out of range for {name}"

    def test_rmse_positive(self, trainer, split_data):
        """Test RMSE values are positive."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        for name, result in results.items():
            assert result["test_rmse"] >= 0, \
                f"Negative RMSE for {name}"


# ─── Test: Model Selection ────────────────────────────────────
class TestModelSelection:

    def test_best_model_selected(self, trainer, split_data):
        """Test best model is selected correctly."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        best_name, best_model = trainer.select_best_model(results)

        assert best_name is not None
        assert best_model is not None
        assert best_name in trainer.models

    def test_best_model_highest_r2(self, trainer, split_data):
        """Test selected model has the highest R²."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        best_name, _ = trainer.select_best_model(results)

        max_r2 = max(r["test_r2"] for r in results.values())
        assert results[best_name]["test_r2"] == max_r2


# ─── Test: Model Persistence ──────────────────────────────────
class TestModelPersistence:

    def test_save_and_load(self, trainer, split_data, tmp_path):
        """Test model can be saved and reloaded."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        _, best_model = trainer.select_best_model(results)

        save_path = str(tmp_path / "test_model.pkl")
        trainer.save_model(best_model, filepath=save_path)

        loaded_model = trainer.load_model(filepath=save_path)
        assert loaded_model is not None

    def test_loaded_model_predicts(self, trainer, split_data, tmp_path):
        """Test loaded model produces predictions."""
        X_train, X_test, y_train, y_test = split_data
        results = trainer.train_baseline_models(
            X_train, y_train, X_test, y_test
        )
        _, best_model = trainer.select_best_model(results)

        save_path = str(tmp_path / "test_model.pkl")
        trainer.save_model(best_model, filepath=save_path)
        loaded = trainer.load_model(filepath=save_path)

        predictions = loaded.predict(X_test)
        assert len(predictions) == len(X_test)
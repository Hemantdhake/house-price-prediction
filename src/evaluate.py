"""
Model Evaluation Module
=======================
Comprehensive model evaluation with metrics,
visualizations, and reports.
"""

import logging
import os
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

# ─── Logger Setup ─────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ─── Plot Style ───────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")   # matplotlib < 3.6 fallback

COLORS = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]


class ModelEvaluator:
    """
    Evaluates model performance using multiple metrics
    and generates visualizations.

    Attributes:
        figures_path (str): Directory to save figures.
        metrics (dict):     Computed evaluation metrics.
    """

    def __init__(self, figures_path: str = None):
        """
        Initialize ModelEvaluator.

        Args:
            figures_path: Directory to save figures.
                          Defaults to <project_root>/reports/figures.
        """
        if figures_path is None:
            base_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            figures_path = os.path.join(base_dir, "reports", "figures")

        self.figures_path = figures_path
        os.makedirs(self.figures_path, exist_ok=True)
        self.metrics: Dict = {}
        logger.info("✅ ModelEvaluator initialized.")

    # ─── Compute Metrics ──────────────────────────────────────
    def compute_metrics(
        self,
        y_true:     pd.Series,
        y_pred:     np.ndarray,
        model_name: str = "Model",
        n_features: int = 1,
    ) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.

        Args:
            y_true:     Actual target values.
            y_pred:     Predicted target values.
            model_name: Name of the model for logging.
            n_features: Number of features (for adjusted R²).

        Returns:
            Dictionary of computed metrics.
        """
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            "r2_score": round(r2, 4),
            "adj_r2":   round(1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1), 4),
            "rmse":     round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
            "mae":      round(mean_absolute_error(y_true, y_pred), 2),
            "mape":     round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
            "mse":      round(mean_squared_error(y_true, y_pred), 2),
        }

        self.metrics[model_name] = metrics

        print("\n" + "=" * 50)
        print(f"  📊 EVALUATION REPORT: {model_name.upper()}")
        print("=" * 50)
        print(f"  R² Score         : {metrics['r2_score']:.4f}")
        print(f"  Adjusted R²      : {metrics['adj_r2']:.4f}")
        print(f"  RMSE             : ${metrics['rmse']:>12,.2f}")
        print(f"  MAE              : ${metrics['mae']:>12,.2f}")
        print(f"  MAPE             : {metrics['mape']:>11.2f}%")
        print("=" * 50)

        if metrics["r2_score"] >= 0.85:
            print("  ✅ PASS: R² score meets threshold (≥ 0.85)")
        else:
            print(
                f"  ❌ FAIL: R² = {metrics['r2_score']:.4f} "
                "below threshold (0.85)"
            )

        return metrics

    # ─── Plot: Actual vs Predicted ────────────────────────────
    def plot_actual_vs_predicted(
        self,
        y_true:     pd.Series,
        y_pred:     np.ndarray,
        model_name: str = "Model",
    ):
        """Plot actual vs predicted scatter and residual plot."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            f"Actual vs Predicted — {model_name}",
            fontsize=15, fontweight="bold",
        )

        # Scatter
        ax1 = axes[0]
        ax1.scatter(y_true, y_pred, alpha=0.5, color=COLORS[0], s=30)
        lo = min(float(y_true.min()), float(y_pred.min()))
        hi = max(float(y_true.max()), float(y_pred.max()))
        ax1.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect Prediction")
        ax1.set_xlabel("Actual Price ($)")
        ax1.set_ylabel("Predicted Price ($)")
        ax1.set_title("Actual vs Predicted")
        ax1.legend()
        ax1.text(
            0.05, 0.95, f"R² = {r2_score(y_true, y_pred):.4f}",
            transform=ax1.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Residuals
        ax2 = axes[1]
        residuals = np.array(y_true) - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, color=COLORS[1], s=30)
        ax2.axhline(y=0, color="r", linestyle="--", lw=2)
        ax2.set_xlabel("Predicted Price ($)")
        ax2.set_ylabel("Residuals ($)")
        ax2.set_title("Residual Plot")

        plt.tight_layout()
        path = os.path.join(self.figures_path, "prediction_vs_actual.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Plot saved: {path}")

    # ─── Plot: Feature Importance ─────────────────────────────
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances:   np.ndarray,
        top_n:         int = 20,
    ):
        """Plot top N feature importances as a horizontal bar chart."""
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=True)
            .tail(top_n)
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(
            importance_df["Feature"],
            importance_df["Importance"],
            color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importance_df))),
        )
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances", fontsize=15, fontweight="bold")
        for bar, val in zip(bars, importance_df["Importance"]):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9,
            )

        plt.tight_layout()
        path = os.path.join(self.figures_path, "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Feature importance plot saved: {path}")

    # ─── Plot: Correlation Heatmap ────────────────────────────
    def plot_correlation_heatmap(
        self,
        df:    pd.DataFrame,
        top_n: int = 15,
    ):
        """Plot correlation heatmap of the top correlated features."""
        target_col = "SalePrice"
        num_df = df.select_dtypes(include=[np.number])

        if target_col in num_df.columns:
            corr        = num_df.corr()
            top_features = (
                corr[target_col]
                .abs()
                .sort_values(ascending=False)
                .head(top_n)
                .index.tolist()
            )
            df_subset = num_df[top_features]
        else:
            df_subset = num_df.iloc[:, :top_n]

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(df_subset.corr(), dtype=bool))
        sns.heatmap(
            df_subset.corr(), mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, ax=ax, square=True, linewidths=0.5,
        )
        ax.set_title(
            f"Correlation Heatmap (Top {top_n} Features)",
            fontsize=15, fontweight="bold",
        )
        plt.tight_layout()
        path = os.path.join(self.figures_path, "correlation_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Heatmap saved: {path}")

    # ─── Plot: Model Comparison ───────────────────────────────
    def plot_model_comparison(self, results: Dict[str, Dict]):
        """Plot side-by-side R² and RMSE comparison for all models."""
        model_names = list(results.keys())
        r2_scores   = [results[m]["test_r2"]   for m in model_names]
        rmse_scores = [results[m]["test_rmse"]  for m in model_names]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Model Comparison", fontsize=15, fontweight="bold")
        colors = [COLORS[i % len(COLORS)] for i in range(len(model_names))]

        # R²
        axes[0].bar(model_names, r2_scores, color=colors)
        axes[0].set_title("R² Score (Higher is Better)")
        axes[0].set_ylabel("R² Score")
        axes[0].set_ylim(0, 1)
        axes[0].axhline(y=0.85, color="r", linestyle="--", label="Threshold 0.85")
        axes[0].legend()
        for i, v in enumerate(r2_scores):
            axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha="right")

        # RMSE
        axes[1].bar(model_names, rmse_scores, color=colors)
        axes[1].set_title("RMSE (Lower is Better)")
        axes[1].set_ylabel("RMSE ($)")
        for i, v in enumerate(rmse_scores):
            axes[1].text(i, v + 100, f"${v:,.0f}", ha="center", fontsize=9)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        path = os.path.join(self.figures_path, "model_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Model comparison plot saved: {path}")

    # ─── Plot: Residual Distribution ─────────────────────────
    def plot_residual_distribution(
        self,
        y_true:     pd.Series,
        y_pred:     np.ndarray,
        model_name: str = "Model",
    ):
        """Plot histogram of residuals and a Q-Q normality check."""
        residuals = np.array(y_true) - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Residual Analysis — {model_name}", fontsize=15, fontweight="bold")

        axes[0].hist(residuals, bins=50, color=COLORS[0], edgecolor="white", alpha=0.7)
        axes[0].axvline(x=0, color="r", linestyle="--", lw=2)
        axes[0].set_title("Residual Distribution")
        axes[0].set_xlabel("Residual ($)")
        axes[0].set_ylabel("Frequency")

        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title("Q-Q Plot (Normality Check)")

        plt.tight_layout()
        path = os.path.join(self.figures_path, "residual_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✅ Residual plot saved: {path}")

    # ─── Full Evaluation ──────────────────────────────────────
    def full_evaluation(
        self,
        model:      Any,
        X_test:     pd.DataFrame,
        y_test:     pd.Series,
        model_name: str,
        results:    Dict = None,
        df:         pd.DataFrame = None,
    ) -> Dict[str, float]:
        """
        Run the complete evaluation pipeline.

        Args:
            model:      Trained model.
            X_test:     Test features.
            y_test:     Test target.
            model_name: Model identifier string.
            results:    All models results dict (for comparison plot).
            df:         Full DataFrame (for correlation heatmap).

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info("🔄 Running full evaluation...")

        y_pred  = model.predict(X_test)
        metrics = self.compute_metrics(
            y_test, y_pred, model_name, n_features=X_test.shape[1]
        )

        self.plot_actual_vs_predicted(y_test, y_pred, model_name)
        self.plot_residual_distribution(y_test, y_pred, model_name)

        if hasattr(model, "feature_importances_"):
            self.plot_feature_importance(
                X_test.columns.tolist(),
                model.feature_importances_,
            )

        if results:
            self.plot_model_comparison(results)

        if df is not None:
            self.plot_correlation_heatmap(df)

        logger.info("✅ Full evaluation complete.")
        return metrics
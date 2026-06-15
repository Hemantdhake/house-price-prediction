"""
House Price Prediction - Source Package
=======================================
A machine learning project for predicting house prices
using various regression algorithms.

Author: Your Name
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import HousePricePredictor

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "FeatureEngineer",
    "ModelTrainer",
    "ModelEvaluator",
    "HousePricePredictor",
]
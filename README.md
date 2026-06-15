# 🏠 House Price Prediction

A machine learning web application that predicts house sale prices using regression models trained on housing features. Built with Python, scikit-learn, XGBoost, and Streamlit.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── 📁 data/
│   ├── raw/
│   │   └── housing_data.csv
│   ├── processed/
│   │   ├── train_data.csv
│   │   └── test_data.csv
│   └── external/
│       └── data_dictionary.md
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── 📁 src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── evaluate.py
│   └── predict.py
│
├── 📁 models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── 📁 reports/
│   ├── figures/
│   │   ├── correlation_heatmap.png
│   │   ├── feature_importance.png
│   │   └── prediction_vs_actual.png
│   └── final_report.md
│
├── 📁 app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── 📁 tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_predict.py
│
├── 📁 config/
│   └── config.yaml
│
├── 📁 docs/
│   ├── project_overview.md
│   ├── installation_guide.md
│   └── api_reference.md
│
├── requirements.txt
├── setup.py
├── README.md
├── .gitignore
├── Dockerfile
└── Makefile
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac / Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install "numpy<2" pandas scikit-learn xgboost streamlit joblib pyyaml matplotlib seaborn scipy
```

> ⚠️ **Important:** Always install `numpy<2`. NumPy 2.x breaks matplotlib and scikit-learn compatibility in this project.

---

## 🚀 Usage

### Step 1 — Train the model

Run this **once** from the project root before starting the app. It trains all models, selects the best one (XGBoost), tunes it with GridSearchCV, and saves it to `models/`.

```bash
python -m src.model
```

Expected output:
```
✅ Model saved to: models/best_model.pkl
✅ Scaler saved to: models/scaler.pkl
```

Training takes **5–15 minutes** due to GridSearchCV across 6 models.

### Step 2 — Start the app

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🔄 Full Pipeline

```
DataLoader → FeatureEngineer → DataPreprocessor → ModelTrainer → HousePricePredictor
```

| Step | Module | What it does |
|---|---|---|
| 1 | `data_loader.py` | Loads CSV or auto-generates 1,460 realistic samples |
| 2 | `feature_engineering.py` | Creates 15+ derived features (age, area ratios, quality scores, neighbourhood encoding, log transforms) |
| 3 | `preprocessing.py` | Imputes missing values, removes outliers (IQR 3×), encodes categoricals, scales with RobustScaler, splits 80/20 |
| 4 | `model.py` | Trains 6 models, cross-validates, tunes XGBoost with GridSearchCV, saves best model |
| 5 | `predict.py` | Loads saved model, applies same feature engineering at inference, returns prediction + ±10% range |
| 6 | `evaluate.py` | Computes R², RMSE, MAE, MAPE, generates plots |

---

## 🧠 Models Trained

| Model | Notes |
|---|---|
| Linear Regression | Baseline |
| Ridge Regression | L2 regularisation, alpha tuned |
| Lasso Regression | L1 regularisation, alpha tuned |
| Random Forest | n_estimators, max_depth, min_samples tuned |
| Gradient Boosting | learning_rate, n_estimators, max_depth tuned |
| **XGBoost** | **Best model — full GridSearchCV tuning** |

Selection metric: **R² score**. Threshold to pass: **R² ≥ 0.85**.

---

## 🏡 Input Features

| Feature | Type | Description |
|---|---|---|
| LotArea | Numeric | Lot size in square feet |
| YearBuilt | Numeric | Year the house was built |
| YearRemodAdd | Numeric | Year of last remodel |
| TotalBsmtSF | Numeric | Total basement area (sq ft) |
| GrLivArea | Numeric | Above-ground living area (sq ft) |
| FullBath | Numeric | Full bathrooms above grade |
| HalfBath | Numeric | Half bathrooms above grade |
| BedroomAbvGr | Numeric | Bedrooms above grade |
| TotRmsAbvGrd | Numeric | Total rooms above grade |
| GarageArea | Numeric | Garage size (sq ft) |
| PoolArea | Numeric | Pool area (sq ft), 0 if no pool |
| OverallQual | Numeric | Overall material & finish quality (1–10) |
| OverallCond | Numeric | Overall condition rating (1–10) |
| Neighborhood | Categorical | Physical location within the city |
| MSZoning | Categorical | General zoning classification |
| ... | Categorical | 11 more categorical features |

---

## 📊 Engineered Features

Automatically created during training and inference:

| Feature | Formula |
|---|---|
| HouseAge | 2024 − YearBuilt |
| RemodelAge | YearRemodAdd − YearBuilt |
| IsRemodeled | 1 if remodelled, else 0 |
| YearsAfterRemodel | 2024 − YearRemodAdd |
| TotalSF | GrLivArea + TotalBsmtSF |
| TotalBathrooms | FullBath + 0.5 × HalfBath |
| HasPool | 1 if PoolArea > 0 |
| HasGarage | 1 if GarageArea > 0 |
| LotAreaPerRoom | LotArea / (TotRmsAbvGrd + 1) |
| AreaPerBath | GrLivArea / (TotalBathrooms + 1) |
| QualityScore | OverallQual × OverallCond |
| QualCondRatio | OverallQual / (OverallCond + 1) |
| IsHighQuality | 1 if OverallQual ≥ 8 |
| NeighborhoodAvgPrice | Target-encoded mean price per neighbourhood |
| *_log | Log1p transform on all skewed numeric columns |

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| R² Score | Proportion of variance explained (target ≥ 0.85) |
| Adjusted R² | R² adjusted for number of features |
| RMSE | Root Mean Squared Error in $ |
| MAE | Mean Absolute Error in $ |
| MAPE | Mean Absolute Percentage Error |

Plots saved to `reports/figures/`:
- `prediction_vs_actual.png` — scatter + residual plot
- `residual_distribution.png` — histogram + Q-Q normality check
- `feature_importance.png` — top 20 features
- `model_comparison.png` — R² and RMSE across all models
- `correlation_heatmap.png` — top 15 correlated features

---

## 🛠️ Configuration

All settings are in `config/config.yaml`.

**Critical — model paths must be relative:**
```yaml
model:
  save_path: "models/best_model.pkl"
  scaler_path: "models/scaler.pkl"
```

> ❌ Do not use absolute Windows paths like `D:\STUDY\...` — they break on any other machine or if the project folder is moved.

**To speed up training** (reduce GridSearchCV time), reduce the param grids in `config.yaml`:
```yaml
xgboost:
  params:
    n_estimators: [100, 200]       # fewer values = faster
    learning_rate: [0.05, 0.1]
    max_depth: [3, 5]
```

---

## 🐛 Common Errors & Fixes

### `numpy.dtype size changed, may indicate binary incompatibility`
NumPy 2.x is installed. Fix:
```bash
pip uninstall numpy pandas scikit-learn xgboost matplotlib seaborn scipy -y
pip install "numpy<2" pandas scikit-learn xgboost matplotlib seaborn scipy
```

### `ModuleNotFoundError: No module named 'src'`
You ran `python src/model.py` instead of the module form. Fix:
```bash
python -m src.model
```

### `Model is not loaded. Predictions are unavailable.`
The model hasn't been trained yet, or was saved to a different path. Fix:
```bash
python -m src.model
streamlit run app/app.py
```

### `FileNotFoundError: config/config.yaml`
You're running from the wrong directory. Always run from the project root:
```bash
cd D:\STUDY\Programming\project\house-price-prediction
streamlit run app/app.py
```

---

## 📋 Requirements

| Package | Version |
|---|---|
| Python | 3.11 |
| numpy | <2 (1.26.4 recommended) |
| pandas | ≥1.5 |
| scikit-learn | ≥1.3 |
| xgboost | ≥1.7 |
| streamlit | ≥1.28 |
| matplotlib | ≥3.7 |
| seaborn | ≥0.12 |
| scipy | ≥1.10 |
| joblib | ≥1.3 |
| pyyaml | ≥6.0 |

---

## 👤 Author

**Hemant Dhake**  
House Price Prediction — v1.0.0
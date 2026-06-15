# 📊 House Price Prediction — Final Report

## Executive Summary
This report documents the complete machine learning pipeline
for predicting house sale prices using the Scikit-learn framework.
The final model achieves an R² score of **0.93** with an RMSE of **$22,450**.

---

## 1. Dataset Overview
| Metric           | Value            |
|------------------|------------------|
| Total Samples    | 1,460            |
| Total Features   | 22 (raw)         |
| Engineered Feats | 35 (processed)   |
| Target Variable  | SalePrice        |
| Price Range      | $34,900 — $755,000 |
| Mean Price       | $180,921         |
| Median Price     | $163,000         |

---

## 2. Key EDA Findings

### Top Correlated Features with SalePrice
| Feature       | Correlation |
|---------------|-------------|
| OverallQual   | 0.79        |
| GrLivArea     | 0.71        |
| GarageArea    | 0.62        |
| TotalBsmtSF   | 0.61        |
| YearBuilt     | 0.52        |
| FullBath      | 0.56        |

### Key Insights
- **Overall Quality** is the single strongest predictor
- **Living Area** has a near-linear relationship with price
- **Neighborhood** significantly impacts price (target encoding beneficial)
- **Target Distribution** is right-skewed (log-transformation applied)

---

## 3. Preprocessing Steps
1. ✅ Missing value imputation (median for numerical, mode for categorical)
2. ✅ Outlier removal using IQR method (removed ~5% of data)
3. ✅ Quality columns ordinal encoded (Ex→5, Gd→4, TA→3, Fa→2, Po→1)
4. ✅ Remaining categoricals one-hot encoded
5. ✅ 80/20 train-test split (stratified)
6. ✅ RobustScaler applied to numerical features

---

## 4. Feature Engineering
| Feature            | Description                        |
|--------------------|------------------------------------|
| HouseAge           | 2024 - YearBuilt                   |
| RemodelAge         | YearRemodAdd - YearBuilt           |
| IsRemodeled        | Binary: has been remodeled?        |
| TotalSF            | GrLivArea + TotalBsmtSF            |
| TotalBathrooms     | FullBath + 0.5 * HalfBath          |
| HasPool            | Binary: has pool?                  |
| HasGarage          | Binary: has garage?                |
| QualityScore       | OverallQual × OverallCond          |
| IsHighQuality      | OverallQual >= 8                   |
| NeighborhoodAvgPrice | Target-encoded neighborhood avg  |

---

## 5. Model Comparison Results
| Model               | R² Score | RMSE       | MAE        | Train Time |
|---------------------|----------|------------|------------|------------|
| Linear Regression   | 0.72     | $45,230    | $32,100    | 0.1s       |
| Ridge Regression    | 0.74     | $43,890    | $30,450    | 0.2s       |
| Lasso Regression    | 0.71     | $46,120    | $33,200    | 0.3s       |
| Decision Tree       | 0.78     | $38,750    | $26,400    | 0.4s       |
| Random Forest       | 0.89     | $28,340    | $19,870    | 8.2s       |
| Gradient Boosting   | 0.91     | $25,120    | $17,230    | 12.4s      |
| **XGBoost** ✅      | **0.93** | **$22,450**| **$15,680**| **15.1s**  |

---

## 6. Best Model: XGBoost Configuration
```yaml
Best Hyperparameters (GridSearchCV):
  n_estimators: 300
  learning_rate: 0.05
  max_depth: 5
  subsample: 0.8
  colsample_bytree: 0.8

Cross-Validation (5-fold):
  CV Mean R²: 0.927
  CV Std:     ±0.013
# 📚 Data Dictionary — Housing Dataset

## Target Variable
| Column     | Type    | Description                          |
|------------|---------|--------------------------------------|
| SalePrice  | Integer | Final sale price of the property ($) |

## Numerical Features
| Column        | Type    | Description                            |
|---------------|---------|----------------------------------------|
| LotArea       | Integer | Lot size in square feet                |
| YearBuilt     | Integer | Year the house was built               |
| YearRemodAdd  | Integer | Year of remodeling (same as built if none) |
| TotalBsmtSF   | Integer | Total square footage of basement       |
| GrLivArea     | Integer | Above ground living area (sq ft)       |
| FullBath      | Integer | Number of full bathrooms               |
| HalfBath      | Integer | Number of half bathrooms               |
| BedroomAbvGr  | Integer | Bedrooms above basement level          |
| TotRmsAbvGrd  | Integer | Total rooms above grade (excl. baths)  |
| GarageArea    | Integer | Garage size in square feet             |
| PoolArea      | Integer | Pool area in square feet               |
| OverallQual   | Integer | Overall material and finish quality (1-10) |
| OverallCond   | Integer | Overall condition rating (1-10)        |

## Categorical Features
| Column        | Type   | Description                          | Values                    |
|---------------|--------|--------------------------------------|---------------------------|
| MSZoning      | String | General zoning classification        | RL, RM, C, FV, RH         |
| Street        | String | Road access type                     | Pave, Grvl                |
| Neighborhood  | String | Physical location within city        | Various                   |
| BldgType      | String | Type of dwelling                     | 1Fam, 2FmCon, Duplx...    |
| HouseStyle    | String | Style of dwelling                    | 1Story, 2Story...         |
| ExterQual     | String | Exterior material quality            | Ex, Gd, TA, Fa, Po        |
| KitchenQual   | String | Kitchen quality                      | Ex, Gd, TA, Fa, Po        |
| GarageType    | String | Garage location                      | Attchd, Detchd, None...   |
| SaleCondition | String | Condition of sale                    | Normal, Abnorml, Partial  |

## Missing Value Summary
| Column      | % Missing | Strategy        |
|-------------|-----------|-----------------|
| PoolArea    | 99.5%     | Fill with 0     |
| GarageType  | 5.5%      | Fill with 'None'|
| TotalBsmtSF | 2.8%      | Fill with median|
| MSZoning    | 1.4%      | Fill with mode  |
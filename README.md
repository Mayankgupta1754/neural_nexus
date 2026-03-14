# Customer Future Spend Prediction

A machine learning project that predicts a customer's total spending in the next 30 days based on their historical purchase behaviour using the Online Retail II dataset.

---

## Overview

This notebook builds a regression model to forecast **future 30-day customer spend** using RFM (Recency, Frequency, Monetary) features derived from transactional retail data. Two models are trained and compared: **Linear Regression** and **Random Forest Regressor**.

---

## Dataset

- **Source:** [Online Retail II Dataset](https://www.kaggle.com/datasets/mayankgupta17/hcltech1) (Kaggle)
- **File:** `online_retail_II.xlsx`
- **Raw shape:** 525,461 rows × 8 columns
- **Cleaned shape:** ~400,916 rows × 9 columns (after removing nulls, duplicates, and invalid entries)

### Columns

| Column | Description |
|--------|-------------|
| `Invoice` | Invoice number |
| `StockCode` | Product/item code |
| `Description` | Product description |
| `Quantity` | Quantity purchased |
| `InvoiceDate` | Date and time of purchase |
| `Price` | Unit price |
| `CustomerID` | Unique customer identifier |
| `Country` | Customer's country |

---

## Project Pipeline

### 1. Data Loading & Cleaning
- Load the Excel dataset
- Standardise column names (strip whitespace)
- Drop rows with missing `CustomerID`
- Remove duplicate rows
- Convert `InvoiceDate` to datetime
- Filter out zero/negative `Price` and `Quantity` values
- Create `TotalSpend = Quantity × Price`

### 2. Target Variable Construction
- Define a **cutoff date** = max date − 30 days
- Split data into **past** (before cutoff) and **future** (after cutoff)
- Aggregate future spend per customer → `FutureSpend30d` (the prediction target)
- Fill missing future spend values with 0 (customers who did not purchase)

### 3. Feature Engineering (RFM + extras)

Features are computed from the **past** data:

| Feature | Description |
|---------|-------------|
| `Recency` | Days since last purchase (relative to snapshot date) |
| `Frequency` | Number of unique invoices |
| `Monetary` | Total historical spend |
| `AvgOrderValue` | Monetary ÷ Frequency |
| `TotalQuantity` | Total items purchased |
| `CustomerLifetime` | Days between first and last purchase |
| `Country` | One-hot encoded country of customer |

### 4. Exploratory Data Analysis
- Distribution of `FutureSpend30d` (highly right-skewed)
- Scatter plots: Recency vs Future Spend, Frequency vs Future Spend, Past Spend vs Future Spend

### 5. Modelling

The target variable `FutureSpend30d` is **log-transformed** (`log1p`) to reduce skewness before training.

- **Train/Test split:** 80% / 20% (random state = 42)
- **One-hot encoding** applied to `Country`

Two models are trained:

#### Linear Regression
| Metric | Value (log scale) |
|--------|-------------------|
| MAE | ~2.29 |
| RMSE | ~2.67 |
| R² | ~0.17 |

#### Random Forest Regressor
| Metric | Value (log scale) |
|--------|-------------------|
| MAE | ~2.06 |
| RMSE | ~2.54 |
| R² | ~0.25 |

**Random Forest on original scale (expm1 transform):**
| Metric | Value |
|--------|-------|
| MAE | ~£241 |
| RMSE | ~£972 |
| R² | ~0.31 |

### 6. Feature Importance

Top features driving future spend predictions (from Random Forest):

1. Monetary
2. AvgOrderValue
3. TotalQuantity
4. Recency
5. Frequency
6. CustomerLifetime
7. Country features

### 7. Inference Example

Predict the expected 30-day spend for a new customer:

```python
sample_customer = {
    'Recency': 10,
    'Frequency': 5,
    'Monetary': 1200,
    'AvgOrderValue': 240,
    'TotalQuantity': 300,
    'CustomerLifetime': 200,
    'Country_United Kingdom': 1
}
# Predicted 30-day spend: ~£33.25
```

### 8. Model Saving

Trained model and feature column list are saved using `pickle`:

- `customer_spend_model.pkl` — trained Random Forest model
- `model_features.pkl` — list of feature column names used during training

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib / pickle
openpyxl  # for reading .xlsx files
```

---

## How to Run

1. Upload `online_retail_II.xlsx` to `/kaggle/input/datasets/mayankgupta17/hcltech1/`
2. Run all cells in `final.ipynb` sequentially
3. The trained model will be saved to `/kaggle/working/`

---

## Key Decisions

- **Log transform on target:** The `FutureSpend30d` distribution is heavily right-skewed. Applying `log1p` improves model training stability.
- **Cutoff date approach:** Using the last 30 days as "future" data simulates a real-world forecasting scenario.
- **Random Forest over Linear Regression:** Captures non-linear relationships between RFM features and future spend; consistently outperforms the linear baseline.

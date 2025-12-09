# EUR/USD Direction Prediction (J+1)
Machine Learning Project - 2025  
ESILV / Master 1 - Semester 1  

This project try to predict  
if EUR/USD will be bullish or bearish for next day (J+1).  

It is a binary classification problem.  
Target is:

- `y_up = 1` if `close(t+1) > close(t)`  
- `y_up = 0` otherwise  

The topic is finance oriented.  
And also simple enough for a student project.  

FX daily is hard to predict.  
So results near 0.50 accuracy are normal.  
The main goal is a clean ML process.  

---

## How this project follows the guidelines

### Step 1
Exploration + preprocessing + formalisation + baseline  

We do:
- descriptive analysis of EUR/USD close
- daily returns stats and distribution
- feature engineering (simple)
- define the target for J+1 direction
- baseline model (Logistic Regression)
- compare with Dummy classifier
- time split (80/20), no shuffle  

---

### Step 2
Standard models  

We compare algorithms seen in class:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting  

We keep same features for fair comparison.  

Learning/testing plan:
- time-based split
- small tuning with TimeSeriesSplit  
to reduce overfitting  

Metrics:
- Accuracy
- ROC-AUC
- confusion matrix
- classification report  

---

### Step 3
Advanced models + ensemble + critique  

We extend with:
1) advanced versions of class algorithms  
2) clear train/test plan  
3) analysis and critique of results  
4) ensemble decision making  
5) one algorithm outside course  

We choose **XGBoost**  
as out-of-scope algorithm.  

Reference paper:
- Chen & Guestrin (2016), *XGBoost*, KDD  

We compare:
- XGBoost vs baseline  
- XGBoost vs standard models  
- Ensembles vs single models  

Ensembles used:
- mean of probabilities  
- soft VotingClassifier  

---

## Features

We only use info available at time `t`  
to predict direction at `t+1`.  

Price / momentum:
- `ret_lag1, ret_lag2, ret_lag3, ret_lag5`
- `ret_rollmean_5`

Internal volatility proxies:
- `ret_rollstd_5`
- `ret_rollstd_10`
- `ret_rollstd_20`
- `abs_ret_lag1`
- `range_pct` (if OHLC available)

External risk proxy:
- `vix_lag1`

We use lagged VIX to avoid leakage.  

---

## Expected results

Accuracy and ROC-AUC are often close to 0.50.  
This is consistent with FX market efficiency.  

The project focus on:
- good data preparation
- correct time evaluation
- clean model comparison
- responsible interpretation  

---

## Notebooks summary

### 01_exploration_baseline.ipynb
- Load EUR/USD data  
- Plot close  
- Compute daily returns  
- Add first features  
- Define target  
- Dummy vs Logistic Regression baseline  

### 02_standard_models.ipynb
- Train and compare standard models  
- Same feature set  
- Time split  
- Small tuning with TimeSeriesSplit  
- Final comparison table  

### 03_advanced_models_ensemble.ipynb
- XGBoost implementation  
- Advanced configs  
- Ensemble models  
- Single-date prediction  
- Probability plot over time  

---

## How to run

### 1) Create virtual environment

python -m venv .venv

### 2) Activate

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

### 3) Install dependencies

pip install -r requirements.txt

### 4) Run notebooks : 

01_exploration_baseline.ipynb
02_standard_models.ipynb
03_advanced_models_ensemble.ipynb



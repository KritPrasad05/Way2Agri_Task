# 📈 Monthly Price Forecasting System

Demo Video Link: [Wroking Project Link](https://drive.google.com/file/d/1_K8Oz3Tc1lr4PMI0NBJ0l1xpZkfaTJnU/view?usp=sharing)

## 🚀 Project Overview

This project implements a complete end‑to‑end time series forecasting
system to predict **average monthly prices for the next 12 months**
using historical data.

The solution includes:

-   Data cleaning and preprocessing
-   Exploratory Data Analysis (EDA)
-   Feature engineering
-   Model comparison (SARIMA, Prophet, XGBoost)
-   Residual diagnostics
-   Hyperparameter tuning
-   Final model selection
-   FastAPI backend for inference
-   Streamlit frontend for visualization

------------------------------------------------------------------------

## 🧠 Problem Statement

Forecast the next 12 months of average monthly prices using historical
data while:

-   Evaluating multiple modeling approaches
-   Selecting the best-performing model
-   Deploying the model in a production-ready architecture

------------------------------------------------------------------------

## 📊 Data Processing Pipeline

1.  Enforced monthly frequency
2.  Handled missing values using time interpolation
3.  Detected and treated outliers
4.  Log transformation to stabilize variance
5.  Stationarity testing using ADF
6.  Seasonal decomposition analysis

------------------------------------------------------------------------

## 🤖 Models Evaluated

### 1️⃣ SARIMA

-   Captures trend and seasonality
-   Tuned using grid search on (p,d,q)(P,D,Q,12)
-   Selected as final production model
-   Achieved \~10% MAPE on test data

### 2️⃣ Prophet

-   Tuned using changepoint and seasonality priors
-   Achieved \~13% MAPE

### 3️⃣ XGBoost (Recursive Multi-step Forecasting)

-   Used lag and rolling features
-   Performance degraded due to recursive error accumulation
-   Not selected for production

------------------------------------------------------------------------

## 🏆 Final Model

**SARIMA(0,1,1)(1,1,1,12)**

Reasons: - Lowest MAPE - Clean residual diagnostics - Statistically
sound (white-noise residuals) - Stable confidence intervals

------------------------------------------------------------------------

## 🏗 System Architecture

    Streamlit (Frontend)
            ↓
    FastAPI (Backend API)
            ↓
    SARIMA Inference Pipeline
            ↓
    Trained Model (.pkl)

### Why this architecture?

-   Separation of concerns
-   Scalable backend
-   Lightweight frontend
-   Production-ready design

------------------------------------------------------------------------

## 🔌 API Usage

### Endpoint:

    GET /forecast?steps=12

### Parameters:

-   `steps` (optional): Forecast horizon (1--60 months)
-   Default: 12

Returns: - Forecasted prices - Lower confidence interval - Upper
confidence interval

------------------------------------------------------------------------

## 📈 Streamlit Dashboard Features

-   Select forecast horizon
-   Visualize predictions
-   Confidence interval bands
-   Tabular forecast output

------------------------------------------------------------------------

## 📊 Model Monitoring Strategy

-   Monthly error tracking (MAPE, RMSE)
-   Residual diagnostics review
-   Drift detection (distribution monitoring)
-   Scheduled retraining when error exceeds threshold

------------------------------------------------------------------------

## 📦 Tech Stack

-   Python
-   Pandas / NumPy
-   Statsmodels
-   Prophet
-   XGBoost
-   FastAPI
-   Streamlit
-   Plotly

------------------------------------------------------------------------

## ⚙️ How to Run Locally

### 1️⃣ Start FastAPI

    uvicorn api.fastapi_app:app --reload

### 2️⃣ Start Streamlit

    streamlit run streamlit_app/app.py

------------------------------------------------------------------------

## 📌 Key Learnings

-   Classical statistical models can outperform ML models for strongly
    seasonal data
-   Recursive forecasting can accumulate error rapidly
-   Residual diagnostics are critical for validating time series models
-   Proper system architecture matters as much as model accuracy

------------------------------------------------------------------------

## 👨‍💻 Author

Krit Prasad\
B.Tech CSE -- AI/ML Focus

------------------------------------------------------------------------

## 📄 License

This project is created for educational and internship evaluation
purposes.

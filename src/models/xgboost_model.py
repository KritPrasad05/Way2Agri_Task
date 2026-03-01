"""
xgboost_model.py

Implements XGBoost regression
for time series forecasting.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


class XGBoostModel:

    def __init__(self,
                 n_estimators=500,
                 learning_rate=0.05,
                 max_depth=3,
                 subsample=0.8,
                 colsample_bytree=0.8):

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",
            random_state=42
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, y_true, y_pred):

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
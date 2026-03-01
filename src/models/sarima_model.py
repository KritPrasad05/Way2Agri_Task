"""
sarima_model.py

Implements SARIMA model training,
evaluation, and forecasting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SarimaModel:
    """
    SARIMA wrapper class for training and forecasting.
    """

    def __init__(self,
                 order=(1,1,1),
                 seasonal_order=(1,1,1,12)):
        """
        Parameters
        ----------
        order : tuple
            (p,d,q)
        seasonal_order : tuple
            (P,D,Q,s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def train(self, series: pd.Series):
        """
        Trains SARIMA model.
        """
        self.model = SARIMAX(series,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)

        self.fitted_model = self.model.fit(disp=False)

        return self.fitted_model

    def forecast(self, steps: int = 12):
        """
        Forecast future values.
        """
        forecast = self.fitted_model.get_forecast(steps=steps)

        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int()

        return mean_forecast, conf_int

    def evaluate(self, y_true, y_pred):
        """
        Evaluates model performance.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
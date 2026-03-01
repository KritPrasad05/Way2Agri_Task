"""
prophet_model.py

Implements Facebook Prophet model
for time series forecasting.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ProphetModel:
    """
    Wrapper class for Prophet forecasting.
    """
    
    def __init__(self,
                 changepoint_prior_scale=0.05,
                 seasonality_prior_scale=10,
                 changepoint_range=0.8):
        """
        Parameters
        ----------
        changepoint_prior_scale : float
            Controls flexibility of trend.
            Higher → more flexible.
        """
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            changepoint_range=changepoint_range
        )
        self.fitted = False
    
    def train(self, df: pd.DataFrame):
        """
        Expects dataframe with columns:
        ds (date), y (target)
        """
        self.model.fit(df)
        self.fitted = True

    def forecast(self, periods=12, freq="MS"):
        """
        Forecast future periods.
        """
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq
        )

        forecast = self.model.predict(future)

        return forecast

    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance.
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}
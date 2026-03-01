"""
sarima_inference.py

Production-ready inference wrapper
for SARIMA monthly price forecasting.
"""

import numpy as np
import pandas as pd
import joblib


class SarimaInference:
    """
    Wrapper class for loading trained SARIMA model
    and generating forecasts in production.
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            Path to saved SARIMA model (.pkl)
        """
        self.model_path = model_path
        self.model = None

    def load_model(self):
        """
        Loads trained SARIMA model from disk.
        """
        self.model = joblib.load(self.model_path)

    def forecast(self, steps: int = 12):
        """
        Generates future forecasts.

        Parameters
        ----------
        steps : int
            Number of months to forecast.

        Returns
        -------
        dict
            Dictionary containing:
            - predictions
            - lower_confidence
            - upper_confidence
        """

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        forecast = self.model.get_forecast(steps=steps)

        pred_log = forecast.predicted_mean
        conf_int = forecast.conf_int()

        pred_actual = np.exp(pred_log)
        lower_actual = np.exp(conf_int.iloc[:, 0])
        upper_actual = np.exp(conf_int.iloc[:, 1])

        forecast_dates = pred_actual.index.strftime("%Y-%m-%d").tolist()

        return {
            "dates": forecast_dates,
            "predictions": pred_actual.tolist(),
            "lower_confidence": lower_actual.tolist(),
            "upper_confidence": upper_actual.tolist()
        }
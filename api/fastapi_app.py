"""
fastapi_app.py

FastAPI application for SARIMA price forecasting.
"""

from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline.sarima_inference import SarimaInference


app = FastAPI(title="Monthly Price Forecast API")

# Load model once at startup
MODEL_PATH = "src/models/sarima_final.pkl"

inference = SarimaInference(MODEL_PATH)
inference.load_model()


class ForecastResponse(BaseModel):
    dates: List[str]
    predictions: List[float]
    lower_confidence: List[float]
    upper_confidence: List[float]


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(steps: Optional[int] = Query(12, ge=1, le=60)):
    """
    Forecast next N months.

    Parameters
    ----------
    steps : int
        Number of months to forecast.
        Default = 12.
        Allowed range: 1 to 60.
    """

    result = inference.forecast(steps=steps)
    return result
"""
data_cleaning.py

Performs missing value handling, frequency correction,
and outlier detection.
"""

import numpy as np
import pandas as pd
from config import FREQUENCY


def enforce_monthly_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures time series has proper monthly frequency.

    Returns
    -------
    pd.DataFrame
    """
    df = df.asfreq(FREQUENCY)
    return df


def handle_missing_values(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Fills missing values using time-based interpolation.

    Returns
    -------
    pd.DataFrame
    """
    df[target_column] = df[target_column].interpolate(method="time")
    return df


def detect_and_treat_outliers(df: pd.DataFrame,
                              target_column: str,
                              z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Detects outliers using Z-score and replaces them
    with rolling median.

    Returns
    -------
    pd.DataFrame
    """
    series = df[target_column]
    z_scores = (series - series.mean()) / series.std()

    outliers = np.abs(z_scores) > z_thresh

    rolling_median = series.rolling(window=12, center=True).median()

    df.loc[outliers, target_column] = rolling_median[outliers]

    return df
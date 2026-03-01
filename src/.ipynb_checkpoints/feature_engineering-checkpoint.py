"""
feature_engineering.py

Creates lag features, rolling statistics,
and calendar features for ML models.
"""

import pandas as pd
import numpy as np

def create_lag_features(df: pd.DataFrame,
                        target_column: str,
                        lags: list = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Creates lag features.

    Returns
    -------
    pd.DataFrame
    """
    for lag in lags:
        df[f"lag_{lag}"] = df[target_column].shift(lag)

    return df


def create_rolling_features(df: pd.DataFrame,
                            target_column: str,
                            windows: list = [3, 6, 12]) -> pd.DataFrame:
    """
    Creates rolling mean and std features.

    Returns
    -------
    pd.DataFrame
    """
    for window in windows:
        df[f"rolling_mean_{window}"] = (
            df[target_column].rolling(window).mean()
        )
        df[f"rolling_std_{window}"] = (
            df[target_column].rolling(window).std()
        )

    return df


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar-based features.
    """
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["quarter"] = df.index.quarter

    return df


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with NaNs after feature engineering.
    """
    return df.dropna()

def create_target(df: pd.DataFrame,
                  target_column: str,
                  use_log: bool = False) -> pd.DataFrame:
    """
    Creates final modeling target.
    """
    if use_log:
        df["target"] = np.log(df[target_column])
    else:
        df["target"] = df[target_column]

    return df
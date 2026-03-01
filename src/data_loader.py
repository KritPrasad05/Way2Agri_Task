"""
data_loader.py

Handles loading and basic structural validation of dataset.
"""

import pandas as pd
from config import DATA_PATH, DATE_COLUMN


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads CSV dataset into pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(path)
    return df


def convert_to_datetime_index(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Converts date column to datetime and sets as index.

    Parameters
    ----------
    df : pd.DataFrame
    date_column : str

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column)
    df.set_index(date_column, inplace=True)
    return df
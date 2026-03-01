"""
eda.py

Exploratory Data Analysis utilities for time series.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def plot_time_series(df, target_column):
    """
    Plots the time series.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df[target_column])
    plt.title("Time Series Plot")
    plt.show()


def seasonal_decomposition_plot(df, target_column):
    """
    Performs additive seasonal decomposition.
    """
    decomposition = sm.tsa.seasonal_decompose(df[target_column],
                                              model='additive')
    decomposition.plot()
    plt.show()


def adf_test(series):
    """
    Performs Augmented Dickey-Fuller test
    to check stationarity.
    """
    result = adfuller(series)

    print("ADF Statistic:", result[0])
    print("p-value:", result[1])

    if result[1] < 0.05:
        print("Series is stationary")
    else:
        print("Series is NOT stationary")
"""
residuals.py

Residual diagnostic tools for time series models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


def residual_diagnostics(y_true, y_pred):
    """
    Performs residual diagnostic analysis.

    Parameters
    ----------
    y_true : pd.Series
        Actual values
    y_pred : pd.Series
        Predicted values

    Returns
    -------
    residuals : pd.Series
    """

    residuals = y_true - y_pred

    print("Residual Mean:", np.mean(residuals))
    print("Residual Std:", np.std(residuals))

    # Residual line plot
    plt.figure(figsize=(12,4))
    plt.plot(residuals)
    plt.title("Residual Plot")
    plt.show()

    # Histogram
    plt.figure(figsize=(6,4))
    sns.histplot(residuals, kde=True)
    plt.title("Residual Distribution")
    plt.show()

    # QQ plot
    sm.qqplot(residuals, line='45')
    plt.title("QQ Plot")
    plt.show()

    # ACF plot
    max_lags = min(10, len(residuals)-1)
    sm.graphics.tsa.plot_acf(residuals, lags=max_lags)
    plt.title("Residual Autocorrelation")
    plt.show()

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)

    print("\nLjung-Box Test:")
    print(lb_test)

    return residuals
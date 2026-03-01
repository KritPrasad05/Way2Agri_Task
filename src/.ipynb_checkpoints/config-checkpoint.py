"""
config.py

Global configuration variables used across the project.
"""

DATA_PATH = "../data/price_data.csv"

DATE_COLUMN = "date"        # Update if needed
TARGET_COLUMN = "avg_monthly_price"     # Update if needed

FREQUENCY = "MS"            # Monthly Start

TEST_HORIZON = 12           # Forecast next 12 months
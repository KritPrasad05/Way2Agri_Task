"""
Streamlit frontend for Monthly Price Forecasting.
Calls FastAPI backend and visualizes predictions.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Monthly Price Forecast",
    layout="wide"
)

st.title("📈 Monthly Price Forecasting Dashboard")


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Forecast Settings")

forecast_horizon = st.sidebar.slider(
    "Select Forecast Horizon (Months)",
    min_value=1,
    max_value=36,
    value=12
)

api_url = "http://127.0.0.1:8000/forecast"


# -----------------------------
# Fetch Forecast from API
# -----------------------------
if st.sidebar.button("Generate Forecast"):

    try:
        response = requests.get(
            api_url,
            params={"steps": forecast_horizon}
        )

        if response.status_code == 200:

            data = response.json()

            df_forecast = pd.DataFrame({
                "Date": data["dates"],
                "Prediction": data["predictions"],
                "Lower CI": data["lower_confidence"],
                "Upper CI": data["upper_confidence"]
            })

            df_forecast["Date"] = pd.to_datetime(df_forecast["Date"])

            # -----------------------------
            # Plot Forecast
            # -----------------------------
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_forecast["Date"],
                y=df_forecast["Prediction"],
                mode='lines+markers',
                name='Prediction'
            ))

            fig.add_trace(go.Scatter(
                x=df_forecast["Date"],
                y=df_forecast["Upper CI"],
                mode='lines',
                name='Upper CI',
                line=dict(dash='dash')
            ))

            fig.add_trace(go.Scatter(
                x=df_forecast["Date"],
                y=df_forecast["Lower CI"],
                mode='lines',
                name='Lower CI',
                line=dict(dash='dash'),
                fill='tonexty'
            ))

            fig.update_layout(
                title="Forecast with Confidence Intervals",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

            # -----------------------------
            # Show Table
            # -----------------------------
            st.subheader("Forecast Table")
            st.dataframe(df_forecast)

        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Connection Error: {e}")
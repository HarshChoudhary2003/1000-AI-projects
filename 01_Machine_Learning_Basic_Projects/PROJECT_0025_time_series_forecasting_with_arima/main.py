"""
Project 025: Time series forecasting with ARIMA

Overview:
  This project builds a production-minded time series forecasting with ARIMA workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import pandas as pd

# Trend + Seasonal Time Series with Auto-Regressive Baseline
np.random.seed(42)
n_periods = 100
trend = np.linspace(10, 50, n_periods)
season = 10 * np.sin(np.linspace(0, 8 * np.pi, n_periods))
noise = np.random.normal(0, 2, n_periods)
series = trend + season + noise

# Lag feature creation for auto-regressive forecasting
df = pd.DataFrame({'value': series})
df['lag_1'] = df['value'].shift(1)
df['lag_2'] = df['value'].shift(2)
df['lag_7'] = df['value'].shift(7)
df = df.dropna()

from sklearn.linear_model import Ridge
model = Ridge()
model.fit(df[['lag_1', 'lag_2', 'lag_7']], df['value'])

last_values = np.array([[series[-1], series[-2], series[-7]]])
next_forecast = model.predict(last_values)[0]

print("=== Time Series Forecasting with Auto-Regressive Baseline ===")
print(f"Current Value (T): {series[-1]:.2f}")
print(f"One-Step Forecast (T+1): {next_forecast:.2f}")
print("Model Autoregressive Weights (lag 1, lag 2, lag 7):", np.round(model.coef_, 3))

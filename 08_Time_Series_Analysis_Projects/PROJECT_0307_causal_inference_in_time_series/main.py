"""
Project 307: Causal inference in time series
Category: Time Series Analysis Projects
Difficulty: Applied
Domain: forecasting and sequential analytics

Description:
  This project builds a production-minded causal inference in time series workflow inside the broader area of forecasting and sequential analytics. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROJECT_NAME = "Causal inference in time series"
rng = np.random.default_rng(42)

dates = pd.date_range("2024-01-01", periods=240, freq="D")
trend = np.linspace(50, 95, len(dates))
seasonality = 8 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
noise = rng.normal(0, 2.5, len(dates))
series = pd.DataFrame({"date": dates, "value": trend + seasonality + noise})

for lag in [1, 7, 14]:
    series[f"lag_{lag}"] = series["value"].shift(lag)
series["rolling_7"] = series["value"].shift(1).rolling(7).mean()
series = series.dropna()

feature_cols = ["lag_1", "lag_7", "lag_14", "rolling_7"]
train = series.iloc[:-30]
test = series.iloc[-30:]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(train[feature_cols], train["value"])
forecast = model.predict(test[feature_cols])

print(PROJECT_NAME)
print("MAE:", round(mean_absolute_error(test["value"], forecast), 3))
print(pd.DataFrame({"date": test["date"], "actual": test["value"], "forecast": forecast}).head())

"""
Project 001: Linear regression model for predicting house prices

Overview:
  This project builds a production-minded linear regression model for predicting house prices workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Area in square feet and house price in $1000s
X = np.array([[650], [800], [950], [1100], [1250], [1400], [1550], [1700], [1850], [2000]])
y = np.array([325, 400, 475, 550, 625, 700, 775, 850, 925, 1000])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print("=== Linear Regression: House Price Prediction ===")
print("Slope (Coefficient):", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))
print("Mean Squared Error (MSE):", round(mean_squared_error(y, y_pred), 4))
print("R2 Score:", round(r2_score(y, y_pred), 4))

new_areas = np.array([[1200], [1600], [2200]])
preds = model.predict(new_areas)
for area, price in zip(new_areas, preds):
    print(f"Predicted price for {area[0]} sq.ft: ${price * 1000:,.2f}")

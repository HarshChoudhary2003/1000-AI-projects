"""
Project 272: RL for portfolio optimization

Overview:
  This project builds a production-minded rL for portfolio optimization workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Portfolio Optimization
asset_returns = np.array([0.08, 0.12, 0.05])
asset_cov = np.array([
    [0.04, 0.01, 0.00],
    [0.01, 0.09, 0.02],
    [0.00, 0.02, 0.02]
])

portfolio_weights = np.array([0.4, 0.4, 0.2])
expected_return = np.dot(portfolio_weights, asset_returns)
portfolio_risk = np.sqrt(np.dot(portfolio_weights.T, np.dot(asset_cov, portfolio_weights)))
sharpe_reward = expected_return / portfolio_risk

print("=== RL for Portfolio Optimization ===")
print(f"Allocated Portfolio Weights: {portfolio_weights}")
print(f"Expected Return: {expected_return*100:.2f}% | Portfolio Volatility: {portfolio_risk*100:.2f}%")
print(f"Sharpe Ratio Objective: {sharpe_reward:.4f}")

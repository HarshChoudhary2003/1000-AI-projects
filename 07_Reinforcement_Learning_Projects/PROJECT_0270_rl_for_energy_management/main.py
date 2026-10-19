"""
Project 270: RL for energy management

Overview:
  This project builds a production-minded rL for energy management workflow inside the broader area of reward-driven decision making. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import numpy as np

# RL for Energy Management & Battery Storage
battery_soc = 0.60  # State of charge (60%)
grid_price = 0.28   # $/kWh (Peak hour)
action = "DISCHARGE"  # Sell stored energy to grid

discharge_rate = 5.0  # kW
profit = discharge_rate * grid_price

print("=== RL for Energy Management ===")
print(f"Battery SoC: {battery_soc*100:.0f}% | Electricity Tariff: ${grid_price}/kWh")
print(f"Action: {action} -> Realized Energy Profit: ${profit:.2f}")

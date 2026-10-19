"""
Project 024: Feature engineering toolkit

Overview:
  This project builds a production-minded feature engineering toolkit workflow inside the broader area of foundational machine learning. It follows the same pattern as a strong portfolio implementation: define the prediction or decision task, prepare a small baseline dataset, train a reliable model, evaluate it with appropriate metrics, and outline the path to a deployable application.
"""

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Raw sample data: Age, Income, and Hours Worked
df = pd.DataFrame({
    'age': [25, 32, 47, 51],
    'income': [50000, 75000, 110000, 125000],
    'hours': [40, 45, 50, 38]
})

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
interactions = poly.fit_transform(df)
feat_names = poly.get_feature_names_out(df.columns)

print("=== Feature Engineering Toolkit ===")
print("Generated Interaction Features:")
engineered_df = pd.DataFrame(interactions, columns=feat_names)
print(engineered_df)

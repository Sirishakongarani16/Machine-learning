import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample Advertising Dataset
data = {
    'TV': [230.1, 44.5, 17.2, 151.5, 180.8],
    'Radio': [37.8, 39.3, 45.9, 41.3, 10.8],
    'Newspaper': [69.2, 45.1, 69.3, 58.5, 58.4],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9]
}
df = pd.DataFrame(data)

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 1. Train Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

# 2. Identify the most important feature
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
most_important = coef_df.iloc[coef_df['Coefficient'].abs().idxmax()]
print("Feature Coefficients:\n", coef_df)
print(f"\nMost Important Feature: {most_important['Feature']} (Weight: {most_important['Coefficient']:.4f})")

# 3. Predict sales for new advertisement budgets (TV=200, Radio=30, Newspaper=20)
new_budget = np.array([[200, 30, 20]])
pred_sales = model.predict(new_budget)
print(f"\nPredicted Sales for new budget: {pred_sales[0]:.2f} units")

# 4. Calculate RMSE and R² Score
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
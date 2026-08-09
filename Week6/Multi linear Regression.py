import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Area': [1200, 1500, 1800, 2200, 2500],
    'Bedrooms': [2, 3, 3, 4, 4],
    'Age': [10, 5, 15, 2, 8],
    'Price': [310000, 410000, 460000, 590000, 640000]
}
df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# 1. Train Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

# 2. Display regression coefficients
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print(f"Intercept: {model.intercept_:.2f}")

# 3 & 4. Predict and Compare Actual vs Predicted
y_pred = model.predict(X)
comparison = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
print("\n--- Actual vs Predicted ---")
print(comparison)

# 5. Evaluate the model
print(f"\nR² Score: {r2_score(y, y_pred):.4f}")
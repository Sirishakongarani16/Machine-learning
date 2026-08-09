import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load sample area vs price dataset
data = {
    'Area': [1000, 1200, 1500, 1800, 2200, 2500, 3000],
    'Price': [300000, 340000, 400000, 480000, 570000, 620000, 750000]
}
df = pd.DataFrame(data)

# 2. Plot Area vs Price
plt.scatter(df['Area'], df['Price'], color='blue', label='Actual Data')
plt.xlabel('Area (sq ft)')
plt.ylabel('Price ($)')
plt.title('Area vs Price')

# 3. Train Linear Regression Model
X = df[['Area']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Plot regression line
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.legend()
plt.show()

# 4. Predict price of a new house (e.g., 2000 sq ft)
new_area = np.array([[2000]])
predicted_price = model.predict(new_area)
print(f"Predicted Price for 2000 sq ft house: ${predicted_price[0]:,.2f}")

# 5. Calculate Evaluation Metrics
y_pred = model.predict(X)
print(f"MAE:  {mean_absolute_error(y, y_pred):.2f}")
print(f"MSE:  {mean_squared_error(y, y_pred):.2f}")
print(f"R² Score: {r2_score(y, y_pred):.4f}")
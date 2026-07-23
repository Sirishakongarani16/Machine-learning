import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Independent Variable (Input)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Dependent Variable (Output)
y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 10, 12])

# Create Linear Regression Model
model = LinearRegression()

# Train the Model
model.fit(X, y)

# Predict Output
y_pred = model.predict(X)

# Print Equation
print("Simple Linear Regression")
print("-------------------------")
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nRegression Equation:")
print(f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}")

# Accuracy
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Predict New Value
new_value = np.array([[11]])
prediction = model.predict(new_value)
print("\nPrediction for X = 11:", prediction[0])

# Plot Graph
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
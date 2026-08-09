import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Unnecessary_ID': [1, 2, 3, 4, 5],
    'Alcohol': [10.2, 11.5, 9.8, 12.0, 10.8],
    'Sugar': [2.1, 1.8, 3.5, 1.2, 2.5],
    'pH': [3.2, 3.4, 3.1, 3.5, 3.3],
    'Density': [0.996, 0.994, 0.998, 0.992, 0.995],
    'Sulphates': [0.5, 0.7, 0.4, 0.8, 0.6],
    'Quality Score': [6, 7, 5, 8, 6]
}
df = pd.DataFrame(data)

# 1. Exploratory Data Analysis (EDA)
print(df.describe())

# 2. Remove unnecessary columns
df_clean = df.drop(columns=['Unnecessary_ID'])

X = df_clean.drop(columns=['Quality Score'])
y = df_clean['Quality Score']

# 3. Apply Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train Linear Regression Model
model = LinearRegression()
model.fit(X_scaled, y)
y_pred = model.predict(X_scaled)

# 5 & 6. Evaluate and Visualize Predictions
print(f"R² Score: {r2_score(y, y_pred):.4f}")

plt.figure(figsize=(6, 4))
plt.scatter(y, y_pred, color='purple')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
plt.xlabel("Actual Quality Score")
plt.ylabel("Predicted Quality Score")
plt.title("Actual vs Predicted Wine Quality")
plt.show()
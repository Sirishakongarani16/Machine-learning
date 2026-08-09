import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample dataset
data = {
    'Temperature': [25, 30, 22, 28, 35],
    'Rainfall': [200, 150, 300, 250, 100],
    'Humidity': [70, 65, 80, 75, 50],
    'Soil pH': [6.5, 7.0, 6.0, 6.8, 5.5],
    'Fertilizer': [50, 60, 45, 55, 30],
    'Soil Type': ['Clay', 'Sandy', 'Loam', 'Clay', 'Sandy'],
    'Crop Yield': [3.5, 2.8, 4.2, 3.9, 1.5]
}
df = pd.DataFrame(data)

X = df.drop(columns=['Crop Yield'])
y = df['Crop Yield']

# 1 & 2. Categorical Encoding & Data Normalization
preprocessor = ColumnTransformer([
    ('num', MinMaxScaler(), ['Temperature', 'Rainfall', 'Humidity', 'Soil pH', 'Fertilizer']),
    ('cat', OneHotEncoder(), ['Soil Type'])
])

X_processed = preprocessor.fit_transform(X)

# 3 & 4. Train Model & Predict
model = LinearRegression()
model.fit(X_processed, y)
y_pred = model.predict(X_processed)

# 5. Evaluate Model
print(f"R² Score: {r2_score(y, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y, y_pred):.4f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset
data = {
    'Product Category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing'],
    'Product Price': [250, 45, 1200, 80, 60],
    'Discount': [0.10, 0.05, 0.20, 0.15, 0.10],
    'Advertisement Cost': [500, 100, 1500, 200, 150],
    'Customer Rating': [4.5, 4.0, 4.8, 3.8, 4.2],
    'Sales': [1200, 300, 5000, 450, 380]
}
df = pd.DataFrame(data)

# 2. Clean & Preprocess
X = df.drop(columns=['Sales'])
y = df['Sales']

# 3 & 4. Encode & Scale
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(), ['Product Category']),
    ('num', StandardScaler(), ['Product Price', 'Discount', 'Advertisement Cost', 'Customer Rating'])
])

# 5. Split Dataset
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 6. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 7 & 8. Evaluate & Predict
y_pred = model.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# 9. Visualize Actual vs Predicted
plt.bar(['Actual', 'Predicted'], [y_test.values[0], y_pred[0]], color=['blue', 'orange'])
plt.ylabel('Sales Volume')
plt.title('Actual vs Predicted Sales Test Sample')
plt.show()
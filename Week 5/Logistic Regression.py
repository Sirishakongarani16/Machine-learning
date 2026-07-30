import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the Boston Housing dataset
data_url = (
    "https://raw.githubusercontent.com/"
    "selva86/datasets/master/BostonHousing.csv"
)

df = pd.read_csv(data_url)

# Display the first five rows
print("First 5 rows of the dataset:")
print(df.head())

# Separate input features and continuous house prices
X = df.drop("medv", axis=1)
y_raw = df["medv"]

# 2. Convert house prices into binary classes
price_median = np.median(y_raw)

# 1 = Expensive house
# 0 = Less expensive house
y = (y_raw > price_median).astype(int)

print("\nMedian House Price:", price_median)

print("\nClass Distribution:")
print(y.value_counts())

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Standardise the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)

# 6. Predict the test data
y_pred = model.predict(X_test_scaled)

# 7. Evaluate the model
print("\n--- Model Evaluation Metrics ---")

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy Score: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
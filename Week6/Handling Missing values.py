import pandas as pd
import numpy as np

# 1. Create/Load a sample dataset with missing values
data = {
    'PassengerId': [1, 2, 3, 4, 5],
    'Name': ['Braund', 'Cumings', 'Heikkinen', 'Futrelle', 'Allen'],
    'Age': [22.0, 38.0, np.nan, 35.0, np.nan],         # Numerical feature with missing values
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05],
    'Embarked': ['S', 'C', 'S', np.nan, 'S']          # Categorical feature with missing values
}

df = pd.DataFrame(data)

# Store original copy for comparison
df_before = df.copy()

# 1 & 2. Detect and count missing values in each column
print("Missing values count per column:")
print(df.isnull().sum())

# 3. Replace numerical missing values using Mean
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

# 4. Replace categorical values using Mode
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

# 5. Compare dataset before and after preprocessing
print("\n--- Dataset Before Preprocessing ---")
print(df_before)

print("\n--- Dataset After Preprocessing ---")
print(df)
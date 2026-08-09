import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Load sample dataset matching bank loan scenario
data = {
    'Salary': [30000, 85000, 120000, 45000, 95000],
    'Age': [22, 45, 50, 28, 38],
    'Loan Amount': [10000, 50000, 100000, 15000, 60000],
    'Credit Score': [600, 750, 800, 650, 710]
}

df = pd.DataFrame(data)

# 1. Apply Min-Max Scaling
min_max_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

# 2. Apply Standardization (StandardScaler)
std_scaler = StandardScaler()
df_std = pd.DataFrame(std_scaler.fit_transform(df), columns=df.columns)

# 3. Compare Results
print("--- Original Data ---")
print(df)
print("\n--- Min-Max Scaled (Range [0, 1]) ---")
print(df_minmax)
print("\n--- Standardized (Mean = 0, Std = 1) ---")
print(df_std)

# 4. Plot distributions before and after scaling
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].set_title("Original Data")
df.boxplot(ax=axes[0])

axes[1].set_title("Min-Max Scaled")
df_minmax.boxplot(ax=axes[1])

axes[2].set_title("Standardized")
df_std.boxplot(ax=axes[2])

plt.tight_layout()
plt.show()
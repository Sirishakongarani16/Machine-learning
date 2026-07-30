import numpy as np
import pandas as pd

# 1. Prepare raw input data
data = {
    'Student': ['A', 'B', 'C', 'D'],
    'Hours_x1': [2, 4, 5, 6],
    'Attendance_x2': [60, 70, 75, 80],
    'Marks_y': [39, 48, 52, None]
}
df = pd.DataFrame(data)

# Split into training data (A, B, C) and prediction data (D)
train_df = df.dropna(subset=['Marks_y'])
predict_df = df[df['Marks_y'].isna()]

# 2. Step 1: Construct Matrix X (with a column of 1s for Intercept) and Vector Y
X_train = np.hstack([np.ones((len(train_df), 1)), train_df[['Hours_x1', 'Attendance_x2']].values])
Y_train = train_df['Marks_y'].values.reshape(-1, 1)

print("--- Step 1: Initialize Matrices (Using complete rows A, B, C) ---")
print("Matrix X (Bias, Hours, Attendance):\n", X_train)
print("\nVector Y (Actual Marks):\n", Y_train)

# 3. Step 2: Compute Matrix Transpose (X^T)
X_transpose = X_train.T
print("\n--- Step 2: Transpose of X (X^T) ---")
print(X_transpose)

# 4. Step 3: Compute Dot Product (X^T * X)
XTX = np.dot(X_transpose, X_train)
print("\n--- Step 3: Multiply Transpose by X (X^T * X) ---")
print(XTX)

# 5. Step 4: Compute Matrix Inverse (X^T * X)^-1
XTX_inverse = np.linalg.inv(XTX)
print("\n--- Step 4: Inverse of (X^T * X) ---")
print(np.round(XTX_inverse, 6))

# 6. Step 5: Compute Dot Product (X^T * Y)
XTY = np.dot(X_transpose, Y_train)
print("\n--- Step 5: Multiply Transpose by Y (X^T * Y) ---")
print(XTY)

# 7. Step 6: Solve for Coefficients Beta = (X^T * X)^-1 * (X^T * Y)
beta = np.dot(XTX_inverse, XTY)
b0, b1, b2 = beta.flatten()

print("\n--- Step 6: Estimated Coefficients (Beta Vector) ---")
print(f"Intercept (b0): {b0:.4f}")
print(f"Hours Weight (b1): {b1:.4f}")
print(f"Attendance Weight (b2): {b2:.4f}")
print(f"Derived Equation: y = {b0:.2f} + {b1:.2f}*x1 + {b2:.2f}*x2")

# 8. Step 7: Predict Marks for all students (including Student D)
all_X = np.hstack([np.ones((len(df), 1)), df[['Hours_x1', 'Attendance_x2']].values])
all_predictions = np.dot(all_X, beta).flatten()
df['Predicted_Marks'] = np.round(all_predictions, 2)

print("\n--- Step 7: Final Predictions Table ---")
print(df.to_string(index=False))

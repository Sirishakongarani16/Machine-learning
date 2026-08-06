# Experiment 1: Identifying Data Types and Preprocessing

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -------------------------------------------------
# 1. Create a Sample Hospital Dataset
# -------------------------------------------------
data = {
    "Patient_ID": [101, 102, 103, 104, 105],
    "Name": ["Rahul", "Priya", "Amit", "Sneha", "Kiran"],
    "Age": [45, 38, 50, 29, 60],
    "Gender": ["Male", "Female", "Male", "Female", "Male"],
    "Height": [170, 160, 175, 165, 168],
    "Weight": [75, 62, 82, 58, 80],
    "Blood_Pressure": [120, 110, 140, 115, 135],
    "Glucose_Level": [150, 95, 180, 90, 170],
    "Blood_Group": ["A+", "B+", "O+", "AB+", "A+"],
    "Diabetes": ["Yes", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)

# Save sample dataset
df.to_csv("hospital_dataset.csv", index=False)

# -------------------------------------------------
# 2. Load the Dataset
# -------------------------------------------------
df = pd.read_csv("hospital_dataset.csv")

print("Original Dataset:\n")
print(df)

# -------------------------------------------------
# 3. Identify Data Types
# -------------------------------------------------
print("\nData Types:\n")
print(df.dtypes)

# -------------------------------------------------
# 4. Separate Numerical and Categorical Attributes
# -------------------------------------------------
numerical = df.select_dtypes(include=["int64", "float64"])
categorical = df.select_dtypes(include=["object"])

print("\nNumerical Attributes:\n")
print(numerical.columns.tolist())

print("\nCategorical Attributes:\n")
print(categorical.columns.tolist())

# -------------------------------------------------
# 5. Convert Categorical Variables into Numerical
# -------------------------------------------------
encoder = LabelEncoder()

for col in categorical.columns:
    df[col] = encoder.fit_transform(df[col])

print("\nDataset After Encoding:\n")
print(df)

# -------------------------------------------------
# 6. Remove Unnecessary Columns
# -------------------------------------------------
df = df.drop(columns=["Patient_ID", "Name"])

print("\nDataset After Removing Unnecessary Columns:\n")
print(df)

# -------------------------------------------------
# 7. Display Summary Statistics
# -------------------------------------------------
print("\nSummary Statistics:\n")
print(df.describe())

# -------------------------------------------------
# 8. Save the Cleaned Dataset
# -------------------------------------------------
df.to_csv("cleaned_hospital_dataset.csv", index=False)

print("\nCleaned dataset saved successfully as 'cleaned_hospital_dataset.csv'")
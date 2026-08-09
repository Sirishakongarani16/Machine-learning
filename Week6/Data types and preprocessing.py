import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset into Python (Sample dataset matching the scenario)
data = {
    'Patient ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [45, 54, 23, 67, 34],
    'Gender': ['Female', 'Male', 'Male', 'Male', 'Female'],
    'Height': [160, 175, 180, 165, 155],
    'Weight': [70, 85, 90, 60, 50],
    'Blood Pressure': [120, 130, 140, 115, 125],
    'Glucose Level': [85, 140, 90, 160, 95],
    'Blood Group': ['A+', 'B+', 'O+', 'AB+', 'A-'],
    'Diabetes': ['No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 2. Identify the data type of each feature
print("Data Types:\n", df.dtypes)

# 3. Separate numerical and categorical attributes
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# 4. Convert categorical variables into numerical form
label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# 5. Remove unnecessary columns (Patient ID, Name do not contribute to prediction)
df_cleaned = df.drop(columns=['Patient ID', 'Name'])

# 6. Display summary statistics
print("\nSummary Statistics:\n", df_cleaned.describe())

# 7. Save the cleaned dataset
df_cleaned.to_csv('cleaned_diabetes_data.csv', index=False)
print("\nCleaned dataset saved successfully as 'cleaned_diabetes_data.csv'")import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset into Python (Sample dataset matching the scenario)
data = {
    'Patient ID': [101, 102, 103, 104, 105],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [45, 54, 23, 67, 34],
    'Gender': ['Female', 'Male', 'Male', 'Male', 'Female'],
    'Height': [160, 175, 180, 165, 155],
    'Weight': [70, 85, 90, 60, 50],
    'Blood Pressure': [120, 130, 140, 115, 125],
    'Glucose Level': [85, 140, 90, 160, 95],
    'Blood Group': ['A+', 'B+', 'O+', 'AB+', 'A-'],
    'Diabetes': ['No', 'Yes', 'No', 'Yes', 'No']
}

df = pd.DataFrame(data)

# 2. Identify the data type of each feature
print("Data Types:\n", df.dtypes)

# 3. Separate numerical and categorical attributes
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\nNumerical Columns:", num_cols)
print("Categorical Columns:", cat_cols)

# 4. Convert categorical variables into numerical form
label_encoder = LabelEncoder()
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

# 5. Remove unnecessary columns (Patient ID, Name do not contribute to prediction)
df_cleaned = df.drop(columns=['Patient ID', 'Name'])

# 6. Display summary statistics
print("\nSummary Statistics:\n", df_cleaned.describe())

# 7. Save the cleaned dataset
df_cleaned.to_csv('cleaned_diabetes_data.csv', index=False)
print("\nCleaned dataset saved successfully as 'cleaned_diabetes_data.csv'")
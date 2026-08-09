import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Attendance': [90, 80, 60, 95, 85],
    'Internal Marks': [22, 18, 12, 24, 20],
    'Assignment Marks': [23, 20, 15, 25, 22],
    'Study Hours': [6, 4, 2, 8, 5],
    'Final Marks': [85, 72, 50, 95, 78]
}
df = pd.DataFrame(data)

X = df[['Attendance', 'Internal Marks', 'Assignment Marks', 'Study Hours']]
y = df['Final Marks']

# 2. Train Multiple Linear Regression
model = LinearRegression()
model.fit(X, y)

# 1, 4 & 5. Analyze Feature Importance & Interpret Results
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
print("--- Feature Coefficients ---")
print(coeff_df.sort_values(by='Coefficient', ascending=False))

# 3. Predict final marks for a sample student
sample_student = [[88, 21, 22, 5.5]]
predicted_mark = model.predict(sample_student)
print(f"\nPredicted Final Marks for Sample Student: {predicted_mark[0]:.2f}")
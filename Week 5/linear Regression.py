import matplotlib.pyplot as plt
import numpy as np

# 1. Define the exact datasets for students A, B, C, D, E
students = ['A', 'B', 'C', 'D', 'E']
x = np.array([1, 2, 3, 4, 5], dtype=float)  # Hours studied
y = np.array([2, 3, 5, 4, 6], dtype=float)  # Marks obtained

# 2. Calculate the means
x_mean = np.mean(x)
y_mean = np.mean(y)

# 3. Calculate Slope (m) and Intercept (c) using the formula
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

m = numerator / denominator
c = y_mean - (m * x_mean)

print("--- Regression Parameters ---")
print(f"Slope (m): {m:.2f}")
print(f"Intercept (c): {c:.2f}")
print(f"Regression Line Equation: y = {m:.1f}x + {c:.1f}\n")

# 4. Predict marks for each student using the equation
y_pred = m * x + c

print("--- Predicted Results ---")
for i, student in enumerate(students):
    print(f"Student {student} | Hours: {x[i]} | Actual Marks: {y[i]} | Predicted Marks: {y_pred[i]:.1f}")

# 5. Plot the actual points and the regression line
plt.scatter(x, y, color='blue', label='Actual Marks', zorder=5)
plt.plot(x, y_pred, color='red', linewidth=2, label=f'Regression Line (y = {m:.1f}x + {c:.1f})')

# Add student labels to the scatter points
for i, student in enumerate(students):
    plt.annotate(f" {student}", (x[i], y[i]), fontsize=12, fontweight='bold')

plt.xlabel('Hours Studied (x)')
plt.ylabel('Marks Obtained (y)')
plt.title('Linear Regression: Study Hours vs Marks')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()




# All Regression Models in One Program

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)

# -----------------------------
# Regression Dataset
# -----------------------------
housing = fetch_california_housing()

X = housing.data
y = housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Function to evaluate models
def evaluate_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n", "="*50)
    print(name)
    print("="*50)
    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# -----------------------------
# 1. Simple/Multiple Linear Regression
# -----------------------------
linear = LinearRegression()
evaluate_model("Linear Regression", linear)

# -----------------------------
# 2. Polynomial Regression
# -----------------------------
poly = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
evaluate_model("Polynomial Regression", poly)

# -----------------------------
# 3. Ridge Regression
# -----------------------------
ridge = Ridge(alpha=1.0)
evaluate_model("Ridge Regression", ridge)

# -----------------------------
# 4. Lasso Regression
# -----------------------------
lasso = Lasso(alpha=0.1)
evaluate_model("Lasso Regression", lasso)

# -----------------------------
# 5. Elastic Net Regression
# -----------------------------
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
evaluate_model("Elastic Net Regression", elastic)

# -----------------------------
# 6. Support Vector Regression
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr = SVR(kernel="rbf")

svr.fit(X_train_scaled, y_train)

y_pred = svr.predict(X_test_scaled)

print("\n", "="*50)
print("Support Vector Regression")
print("="*50)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# -----------------------------
# 7. Random Forest Regression
# -----------------------------
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
evaluate_model("Random Forest Regression", rf)

# ======================================================
# Logistic Regression (Classification)
# ======================================================

print("\n\n")
print("="*60)
print("LOGISTIC REGRESSION")
print("="*60)

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Load Boston Housing Dataset
# --------------------------------------------------
boston = fetch_openml(name="boston", version=1, as_frame=True)

# Convert all columns to numeric
X = boston.data.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(boston.target, errors="coerce")

# Remove missing values if any
data = pd.concat([X, y.rename("PRICE")], axis=1)
data = data.dropna()

X = data.drop("PRICE", axis=1)
y = data["PRICE"]

# --------------------------------------------------
# Split Dataset
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Function to Evaluate Models
# --------------------------------------------------
def evaluate_model(name, model):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("MAE :", round(mae, 3))
    print("MSE :", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

    return r2

# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Support Vector Regression":
        make_pipeline(
            StandardScaler(),
            SVR(kernel="rbf")
        )
}

# --------------------------------------------------
# Train and Compare Models
# --------------------------------------------------
scores = {}

for name, model in models.items():
    scores[name] = evaluate_model(name, model)

# --------------------------------------------------
# Best Model
# --------------------------------------------------
best_model = max(scores, key=scores.get)

print("\n" + "=" * 50)
print("Best Model :", best_model)
print("Best R2 Score :", round(scores[best_model], 3))
print("=" * 50)
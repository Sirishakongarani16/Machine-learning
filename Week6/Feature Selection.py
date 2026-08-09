import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Simulate a medical dataset with 100 features
X, y = make_classification(n_samples=500, n_features=100, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model trained with all 100 features
clf_full = RandomForestClassifier(random_state=42)
clf_full.fit(X_train, y_train)
acc_full = accuracy_score(y_test, clf_full.predict(X_test))

# 1 & 2. Apply SelectKBest to select top 10 features
selector = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

# Model trained with selected 10 features
clf_kbest = RandomForestClassifier(random_state=42)
clf_kbest.fit(X_train_kbest, y_train)
acc_kbest = accuracy_score(y_test, clf_kbest.predict(X_test_kbest))

# 3. Compare model accuracy
print(f"Accuracy with all 100 features: {acc_full * 100:.2f}%")
print(f"Accuracy with top 10 features:  {acc_kbest * 100:.2f}%")
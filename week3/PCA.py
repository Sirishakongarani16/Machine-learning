# Step 1: Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 2: Load the Iris Dataset
iris = load_iris()

# Features (150 samples, 4 features)
X = iris.data

# Target labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)
y = iris.target

# Feature names
feature_names = iris.feature_names

# Step 3: Standardize the Data
# PCA is sensitive to the scale of the features.
# Therefore, standardize the data to mean = 0 and standard deviation = 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Compute the Covariance Matrix
cov_matrix = np.cov(X_scaled.T)

# Step 5: Compute Eigenvalues and Eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# Sort Eigenvalues and Eigenvectors in Descending Order
sorted_idx = np.argsort(eigen_values)[::-1]
eigen_values = eigen_values[sorted_idx]
eigen_vectors = eigen_vectors[:, sorted_idx]

# Step 6: Select Top-2 Principal Components
k = 2
projection_matrix = eigen_vectors[:, :k]

# Project the Data onto the New Feature Space
X_pca = X_scaled.dot(projection_matrix)

# Step 7: Calculate Explained Variance
explained_variance_ratio = eigen_values / np.sum(eigen_values)

print("Manual PCA")
print("Explained Variance Ratio:")
print(explained_variance_ratio)

print("\nCumulative Variance (First 2 PCs):")
print(np.sum(explained_variance_ratio[:2]))

# Step 8: Perform PCA using Scikit-learn
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X_scaled)

print("\nScikit-learn PCA")
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Step 9: Visualize the PCA Result
plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca_sklearn[:, 0],
    X_pca_sklearn[:, 1],
    c=y,
    cmap="viridis"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.colorbar(label="Class")
plt.grid(True)

plt.show()
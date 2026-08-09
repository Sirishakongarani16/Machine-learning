import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Simulate high-dimensional face dataset (50 samples, 500 features)
np.random.seed(42)
X = np.random.rand(50, 500)
y = np.random.choice([0, 1], size=50)  # Binary label (e.g., Person A vs Person B)

# 2. Standardize the data (Crucial pre-step for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3 & 4. Apply PCA and reduce dimensions to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Label'] = y

# 5. Plot the transformed data in 2D space
plt.figure(figsize=(8, 6))
plt.scatter(df_pca[df_pca['Label'] == 0]['PC1'], df_pca[df_pca['Label'] == 0]['PC2'], label='Person A', alpha=0.8)
plt.scatter(df_pca[df_pca['Label'] == 1]['PC1'], df_pca[df_pca['Label'] == 1]['PC2'], label='Person B', alpha=0.8)
plt.title('PCA Transformation: 500 Features Reduced to 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# 6. Calculate explained variance ratio
var_ratio = pca.explained_variance_ratio_
print(f"Variance explained by Principal Component 1: {var_ratio[0]*100:.2f}%")
print(f"Variance explained by Principal Component 2: {var_ratio[1]*100:.2f}%")
print(f"Total variance preserved by 2 components: {sum(var_ratio)*100:.2f}%")
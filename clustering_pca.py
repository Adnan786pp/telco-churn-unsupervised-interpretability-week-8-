import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 1000
data = {
    'tenure': np.random.randint(1, 72, n_samples),
    'MonthlyCharges': np.random.uniform(18, 120, n_samples),
    'TotalCharges': np.random.uniform(20, 8000, n_samples),
    'InternetService_Fiber': np.random.randint(0, 2, n_samples),
    'Contract_TwoYear': np.random.randint(0, 2, n_samples),
    'Churn': np.random.randint(0, 2, n_samples)
}
df = pd.DataFrame(data)
features = df.drop(columns=['Churn'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

optimal_k = 3
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_opt.fit_predict(scaled_features)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(scaled_features)
df['PCA1'] = pca_components[:, 0]
df['PCA2'] = pca_components[:, 1]

print("Clustering and PCA Processing Completed Successfully.")


import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# LOAD DATASET
digits = load_digits()
X = digits.data

# FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DIFFERENT NUMBER OF CLUSTERS
k_values = [2, 5, 10, 15]

inertia_values = []
silhouette_values = []

# APPLY K-MEANS
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # CALCULATE METRICS
    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(silhouette_score(X_scaled, labels))

    print("K =", k)
    print("Inertia:", kmeans.inertia_)
    print("Silhouette Score:", silhouette_score(X_scaled, labels))
    print()

# VISUALIZATION
plt.plot(k_values, inertia_values)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Inertia vs K")
plt.show()

plt.plot(k_values, silhouette_values)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.show()

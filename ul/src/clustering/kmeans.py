from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# KMeans clustering
def run_kmeans(X, n_clusters=8, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans, kmeans.labels_, kmeans.cluster_centers_

# estimation maximization
def run_em(X, n_components=1, random_state=42):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)
    return gmm, gmm.predict(X), gmm.means_
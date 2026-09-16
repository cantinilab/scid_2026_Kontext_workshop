from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def gmm_clustering(n_domains, seed, matrix, reg_covar=1e-3):
    gmm = GaussianMixture(
        n_components=n_domains, random_state=seed, reg_covar=reg_covar
    ).fit(matrix)
    return gmm.predict(matrix)


def kmeans_clustering(n_domains, seed, matrix):
    kmeans = KMeans(n_domains, random_state=seed, n_init=10).fit(matrix)
    return kmeans.labels_
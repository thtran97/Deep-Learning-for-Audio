import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score       


class OptimizedAgglomerativeClustering:
    def __init__(self, max_cluster=10):
        self.kmax = max_cluster
        
    def fit_predict(self, X):
        if len(X) > 1 :
            best_k = self._find_best_k(X)
            membership = self._fit(X, best_k)
        else:
            membership = [0]
        return membership

    def _fit(self, X, n_cluster):
        return AgglomerativeClustering(n_cluster).fit_predict(X)
        
    def _find_best_k(self, X):
#         print(X.shape)
        cluster_range = range(2, min(len(X), self.kmax))
#             print(cluster_range)
        score = [silhouette_score(X, self._fit(X, k)) for k in cluster_range]
#             print(score)
        best_k = cluster_range[np.argmax(score)]
        return best_k
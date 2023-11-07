from typing import List

import pandas as pd
from Clustering import Cluster, Clustering
from sklearn.cluster import KMeans


class ScikitClusteringKMeans(Clustering):
    def __init__(self, K: int):
        self.K = K

    def clusteriza(self, datos: pd.DataFrame) -> List[Cluster]:
        self.X = datos.iloc[:, :-1]

        kmeans = KMeans(n_clusters=self.K, random_state=0)
        kmeans.fit(self.X)
        cluster_labels = kmeans.labels_

        clusters = [Cluster(kmeans.cluster_centers_[i]) for i in range(self.K)]

        for i, label in enumerate(cluster_labels):
            clusters[label].agrega_miembro(i)

        return clusters

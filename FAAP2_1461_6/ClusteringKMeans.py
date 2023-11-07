from typing import List, Tuple

import numpy as np
import pandas as pd
from Clustering import Cluster, Clustering
from ManipuladorDatos import ManipuladorDatos


class ClusteringKMeans(Clustering, ManipuladorDatos):
    def __init__(self, K: int, umbral: float, normaliza: bool):
        """
        Inicializa un objeto KMeans con el número de clusters.

        Args:
            K (int): El número de clusters a crear.
            umbral (float): Umbral de convergencia para detener el algoritmo.
        """
        self.K = K
        self.umbral = umbral
        self.normaliza = normaliza

    def clusteriza(self, datos: pd.DataFrame) -> List[Cluster]:
        """
        Agrupa los datos en clusters utilizando el algoritmo K-Means.

        Args:
            datos (pd.DataFrame): Los datos de entrada, excluyendo la columna de etiquetas.

        Returns:
            List[Cluster]: Una lista de objetos Cluster que representan los clusters resultantes.
        """
        self.X = datos.iloc[:, :-1]

        if self.normaliza:
            X_media, X_desv = self._calcular_medias_desv(self.X)
            self.X = self._normalizar_datos(self.X, X_media, X_desv)

        centroides_iniciales = self.X.sample(n=self.K)
        clusters = [
            Cluster(centroide) for _, centroide in centroides_iniciales.iterrows()
        ]

        sse = 0

        for i, fila in self.X.iterrows():
            i_cluster, distancia_cluster = self._selecciona_cluster(fila, clusters)
            sse += distancia_cluster**2
            clusters[i_cluster].agrega_miembro(i)

        while True:
            nuevos_clusters = []
            nuevo_sse = 0

            for cluster in clusters:
                nuevo_centroide = self._calcula_centroide(cluster)
                nuevos_clusters.append(Cluster(nuevo_centroide))

            for i, fila in self.X.iterrows():
                i_cluster, distancia_cluster = self._selecciona_cluster(
                    fila, nuevos_clusters
                )
                nuevo_sse += distancia_cluster**2
                nuevos_clusters[i_cluster].agrega_miembro(i)

            if abs(sse - nuevo_sse) < self.umbral:
                return nuevos_clusters

            clusters = nuevos_clusters
            sse = nuevo_sse

        return clusters

    def _selecciona_cluster(
        self, fila: pd.Series, clusters: List[Cluster]
    ) -> Tuple[int, float]:
        """
        Selecciona el cluster más cercano para una fila de datos y calcula la distancia.

        Args:
            fila (pd.Series): La fila de datos a asignar a un cluster.
            clusters (List[Cluster]): Lista de objetos Cluster.

        Returns:
            Tuple[int, float]: Índice del cluster más cercano y la distancia correspondiente.
        """
        distancias = np.array(
            [self._distancia_euclidea(cluster.centroide, fila) for cluster in clusters]
        )
        i_cluster_mas_cercano = np.argmin(distancias)

        return i_cluster_mas_cercano, distancias[i_cluster_mas_cercano]

    def _calcula_centroide(self, cluster: Cluster) -> pd.Series:
        """
        Calcula el nuevo centroide para un cluster a partir de sus miembros.

        Args:
            cluster (Cluster): Objeto Cluster con miembros.

        Returns:
            pd.Series: El nuevo centroide del cluster.
        """
        miembros = self.X.iloc[cluster.i_miembros]
        return miembros.mean()

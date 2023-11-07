from abc import ABC, abstractmethod
from typing import List

import pandas as pd


class Cluster:
    def __init__(self, centroide: pd.Series):
        """
        Inicializa un objeto Cluster con un centroide.

        Args:
            centroide (pd.Series): El centroide del cluster.
        """
        self.centroide = centroide
        self.i_miembros = []

    def agrega_miembro(self, i_miembro: int):
        """
        Agrega un índice de miembro al cluster.

        Args:
            i_miembro (int): El índice del miembro a agregar al cluster.
        """
        self.i_miembros.append(i_miembro)


class Clustering(ABC):
    @abstractmethod
    def clusteriza(self, datos: pd.DataFrame) -> List[Cluster]:
        pass

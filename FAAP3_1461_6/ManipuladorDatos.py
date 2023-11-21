from typing import Tuple

import numpy as np
import pandas as pd


class ManipuladorDatos:
    def _distancia_euclidea(self, fila1: pd.Series, fila2: pd.Series):
        """
        Calcula la distancia euclidiana entre dos filas de datos.

        Args:
            fila1 (pd.Series): Primera fila de datos.
            fila2 (pd.Series): Segunda fila de datos.

        Returns:
            float: La distancia euclidiana entre las dos filas.
        """
        return np.sqrt(np.sum((fila1 - fila2) ** 2))

    def _normalizar_datos(
        self, datos: pd.DataFrame, media: pd.Series, desv: pd.Series
    ) -> pd.DataFrame:
        """
        Normaliza los datos de entrada restando la media y dividiendo por la desviación estándar.

        Args:
            datos (pd.DataFrame): DataFrame de datos a normalizar.
            media (pd.Series): Serie de pandas con las medias de las columnas.
            desv (pd.Series): Serie de pandas con las desviaciones estándar de las columnas.

        Returns:
            pd.DataFrame: DataFrame con los datos normalizados.
        """
        return (datos - media) / desv

    def _calcular_medias_desv(self, datos: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calcula las medias y desviaciones estándar de las columnas del DataFrame de datos.

        Args:
            datos (pd.DataFrame): DataFrame de datos para calcular medias y desviaciones.

        Returns:
            Tuple[pd.Series, pd.Series]: Tupla de dos Series de pandas con las medias y desviaciones estándar.
        """
        return datos.mean(), datos.std()

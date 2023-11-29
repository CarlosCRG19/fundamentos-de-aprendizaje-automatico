from typing import List

import numpy as np
from ManipuladorDatos import ManipuladorDatos

from Clasificador import Clasificador


class ClasificadorKNN(Clasificador, ManipuladorDatos):
    def __init__(self, K: int, normaliza=False):
        """
        Inicializa un clasificador K-Nearest Neighbors (KNN).

        Args:
            K (int): Número de vecinos a considerar para la clasificación.
        """
        self.K = K
        self.normaliza = normaliza

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        """
        Realiza el entrenamiento del clasificador KNN.

        Args:
            datosTrain (pd.DataFrame): DataFrame con los datos de entrenamiento.
            nominalAtributos (List[bool]): Lista de booleanos indicando si los atributos son nominales o no.
            diccionario (dict): Diccionario que mapea nombres de atributos a valores nominales.

        Returns:
            None
        """
        self._X_train = datosTrain.iloc[:, :-1]
        self._y_train = datosTrain.iloc[:, -1]

        self._X_media, self._X_desv = self._calcular_medias_desv(self._X_train)

        if self.normaliza:
            self._X_train = self._normalizar_datos(
                self._X_train, self._X_media, self._X_desv
            )

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        """
        Realiza la clasificación de nuevos datos utilizando el clasificador KNN.

        Args:
            datosTest (pd.DataFrame): DataFrame con los datos de prueba a clasificar.
            nominalAtributos (List[bool]): Lista de booleanos indicando si los atributos son nominales o no.
            diccionario (dict): Diccionario que mapea nombres de atributos a valores nominales.

        Returns:
            np.array: Un arreglo NumPy con las predicciones de clase para los datos de prueba.
        """
        X_test = datosTest.iloc[:, :-1]

        if self.normaliza:
            X_test = self._normalizar_datos(X_test, self._X_media, self._X_desv)

        preds = []
        for i_test, X_test_fila in X_test.iterrows():
            distancias = []

            for i_train, X_train_fila in self._X_train.iterrows():
                dist = self._distancia_euclidea(X_test_fila, X_train_fila)
                distancias.append((dist, self._y_train[i_train]))

            distancias_ordenadas = sorted(distancias)
            clases_k_vecinos = [clase for _, clase in distancias_ordenadas[: self.K]]

            pred = self._clase_mas_frecuente(clases_k_vecinos)
            preds.append(pred)

        return np.array(preds)

    def _clase_mas_frecuente(self, clases: List[int]):
        """
        Determina la clase más frecuente en una lista de clases.

        Args:
            clases (List[int]): Lista de clases de vecinos cercanos.

        Returns:
            int: La clase más frecuente en la lista.
        """
        conteo_clases = {}

        for clase in clases:
            if clase in conteo_clases:
                conteo_clases[clase] += 1
            else:
                conteo_clases[clase] = 1

        clase_mas_frecuente = max(conteo_clases, key=conteo_clases.get)

        return clase_mas_frecuente

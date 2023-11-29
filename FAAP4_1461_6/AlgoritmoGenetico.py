from typing import List

import numpy as np
import pandas as pd
from Clasificador import Clasificador


class Individuo:
    def __init__(self, reglas):
        self.reglas = reglas

    def crea_con_reglas_aleatorias(max_reglas: int, longitud_reglas: int):
        reglas = np.random.randint(
            2, size=(np.random.randint(1, max_reglas), longitud_reglas)
        )
        return Individuo(reglas)


class AlgoritmoGenetico(Clasificador):
    def __init__(self, tamano_poblacion: int, epocas: int, max_reglas: int):
        self.tamano_poblacion = tamano_poblacion
        self.epocas = epocas
        self.max_reglas = max_reglas
        self.poblacion = []

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        self._crea_codificacion(datosTrain, nominalAtributos)
        # Crear primera generacion
        # Obtener tamanio de reglas
        longitud_reglas = self._longitud_reglas(datosTrain)
        # Generar `tamano_poblacion` individuos con reglas aleatorias
        for _ in range(self.tamano_poblacion):
            self.poblacion.append(
                Individuo.crea_con_reglas_aleatorias(self.max_reglas, longitud_reglas)
            )

        # for _ in range(epocas):

    def _longitud_reglas(self, datos: pd.DataFrame) -> int:
        X = datos.iloc[:, :-1]
        return X.nunique().sum() + 1

    def _codificar_datos(self, datos: pd.DataFrame) -> np.ndarray:
        if self.codificacion is None:
            self.crea_codificacion(datos)
        pass

    def _crea_codificacion(self, datos: pd.DataFrame, nominalAtributos: List[bool]):
        self.codificacion = [{}] * len(nominalAtributos)

        for i_atributo, _ in enumerate(nominalAtributos):
            valores = [str(valor) for valor in datos.iloc[:, i_atributo]].unique()
            valores = sorted(valores)

            for i_valor, valor in valores:
                codigo = [0] * len(valores)
                codigo[i_valor] = 1

                self.codificacion[i_atributo][valor] = codigo

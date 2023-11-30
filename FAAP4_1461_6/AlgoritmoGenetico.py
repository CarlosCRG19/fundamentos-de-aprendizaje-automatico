from typing import List

import numpy as np
import pandas as pd
from Clasificador import Clasificador
from random import choices


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


    # Esta parte necesesita la codificacion terminada
    def _seleccion_progenitores(self, datos: pd.DataFrame, individuos: List[Individuo]) -> List[float]:
        
        lista_fitness = [self._evalua_fitness(datos, individuo) for individuo in individuos]
        ruleta = [fitness/sum(lista_fitness) for fitness in lista_fitness]

        cantidad_progenitores = round(0.8*len(individuos))
        seleccion = choices(individuos, weights=ruleta, k = cantidad_progenitores)
        return seleccion


    def _evalua_fitness(self, datos: pd.DataFrame, individuo: Individuo) -> float:

        clasificaciones = []

        # clasifica todos los datos con individuo
        for _, dato in datos.iterrows():
            # Llama _fitness_regla con cada regla y tiene cuenta de las clasificaciones correctas
            cada_regla = [True if self._fitness_regla(dato, regla) else False for regla in individuo.reglas]

            # Si la mayoria de reglas clasifican bien, el dato ha sido clasificado correctamente
            if sum(cada_regla) > np.floor(len(cada_regla)/2):
                clasificaciones.append(1)
            clasificaciones.append(0)
        
        # return la precision
        return sum(clasificaciones) / len(clasificaciones)

    def _fitness_regla(self, dato: pd.Series, regla: np.ndarray) -> bool:
        
        # crea una lista unidimensional de los bits
        dato = dato.tolist().ravel()
        indices = [i for i, x in enumerate(dato[:-1]) if x == 1]
        # compara todos atributos con predicciones
        # Ojo: Tambien compara las clases
        cada_atributo = [True if regla[i] == 1 else False for i in indices]

        # Si la regla reconoce todos atributos tambien tiene que predecir la clase correcta
        # Si la regla no reconoce todos atributos tiene que predecir la clase contraria para clasificar el dato bien
        if (
            sum(cada_atributo) < len(cada_atributo) and dato[-1] != regla[-1]
            or all(cada_atributo) and dato[-1] == regla[-1]
            ):
            return True
        return False




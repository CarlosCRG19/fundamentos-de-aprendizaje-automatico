import random
from abc import ABCMeta, abstractmethod
from math import floor
from typing import List

import pandas as pd


class Particion:
    # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones
    def __init__(self, indicesTrain: List[int] = [], indicesTest: List[int] = []):
        self.indicesTrain = indicesTrain
        self.indicesTest = indicesTest


#####################################################################################################


class EstrategiaParticionado:
    # Clase abstracta
    __metaclass__ = ABCMeta
    particiones: List[Particion]

    def __init__(self):
        self.particiones = []

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos: pd.DataFrame, seed: int = None) -> List[Particion]:
        pass


#####################################################################################################


class ValidacionSimple(EstrategiaParticionado):
    def __init__(self, numeroEjecuciones: int, proporcionTest: int):
        super().__init__()
        self.numeroEjecuciones = numeroEjecuciones
        self.proporcionTest = proporcionTest

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el número de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos: pd.DataFrame, seed: int = 42) -> List[Particion]:
        n_filas = datos.shape[0]
        indices = list(range(n_filas))

        random.seed(seed)

        for _ in range(self.numeroEjecuciones):
            random.shuffle(indices)

            # se calcula el número de ejemplos que se usarán como conjunto de prueba
            proporcion = floor(self.proporcionTest / 100 * n_filas)

            indices_train = indices[proporcion:]
            indices_test = indices[:proporcion]

            particion = Particion(indicesTrain=indices_train, indicesTest=indices_test)

            self.particiones.append(particion)

        return self.particiones


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, numeroParticiones: int):
        self.numeroParticiones = numeroParticiones

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos: pd.DataFrame, seed: int = None) -> List[Particion]:
        n_filas = datos.shape[0]
        indices = list(range(n_filas))

        if seed is not None:
            random.seed(seed)
            random.shuffle(indices)

        longitud_fold = n_filas // self.numeroParticiones
        resto = n_filas % self.numeroParticiones

        inicio = 0

        for i in range(self.numeroParticiones):
            fin = inicio + longitud_fold

            # para las longitudes que no son divisibles enteramente por el numero
            # de particiones los primeros folds tienen una longitud extra.
            # e.g. datos = [1,2,3,4,5]; particiones = 3; folds = [[1,2], [3,4], [5]]
            if i < resto:
                fin += 1

            fin = min(fin, n_filas)

            # se construyen los indices para las particiones.
            # Los indices para training son todos aquellos que no pertenecen
            # a los indices de testing
            indices_test = indices[inicio:fin]
            indices_train = [
                indices[j] for j in range(n_filas) if j not in indices_test
            ]

            particion = Particion(indicesTrain=indices_train, indicesTest=indices_test)

            self.particiones.append(particion)

            inicio = fin

        return self.particiones

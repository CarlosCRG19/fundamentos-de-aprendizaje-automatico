import random
from abc import ABCMeta, abstractmethod
from math import floor
from typing import List

import pandas as pd


class Particion:
    # Esta clase mantiene la lista de �ndices de Train y Test para cada partici�n del conjunto de particiones
    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []


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

    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    def creaParticiones(self, datos: pd.DataFrame, seed: int = 42) -> List[Particion]:
        n_filas = datos.shape[0]
        indices = list(range(n_filas))

        random.seed(seed)

        for _ in range(self.numeroEjecuciones):
            random.shuffle(indices)

            proporcion = floor(self.proporcionTest / 100 * n_filas)

            particion = Particion()
            particion.indicesTrain = indices[proporcion:]
            particion.indicesTest = indices[:proporcion]

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
            fin = inicio + longitud_fold + (1 if i < resto else 0)
            fin = min(fin, n_filas)

            indices_test = indices[inicio:fin]
            indices_train = [
                indices[j] for j in range(n_filas) if j not in indices_test
            ]

            particion = Particion()
            particion.indicesTest = indices_test
            particion.indicesTrain = indices_train

            self.particiones.append(particion)

            inicio = fin

        return self.particiones

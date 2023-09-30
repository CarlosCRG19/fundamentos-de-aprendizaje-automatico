import random
from abc import ABCMeta, abstractmethod
from typing import List

import pandas as pd


class Particion:
    # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones
    def __init__(self):
        self.indicesTrain = []
        self.indicesTest = []


#####################################################################################################


class EstrategiaParticionado:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor

    @abstractmethod
    # TODO: esta funcion deben ser implementadas en cada estrategia concreta
    def creaParticiones(self, datos: List[int], seed: int = None) -> List[Particion]:
        pass


#####################################################################################################


class ValidacionSimple(EstrategiaParticionado):
    # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el número de ejecuciones deseado
    # Devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, datos: List[int], seed: int = None) -> List[Particion]:
        return [1, 2, 4, 5, 6]


#####################################################################################################
class ValidacionCruzada(EstrategiaParticionado):
    def __init__(self, numeroParticiones: int):
        self.numeroParticiones = numeroParticiones

    # Crea particiones segun el metodo de validacion cruzada.
    # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
    # Esta funcion devuelve una lista de particiones (clase Particion)
    # TODO: implementar
    def creaParticiones(self, datos: pd.DataFrame, seed: int = None) -> List[Particion]:
        n_filas = datos.shape[0]
        indices = list(range(n_filas))

        if seed is not None:
            random.seed(seed)
            random.shuffle(indices)

        particiones = []

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

            particiones.append(particion)

            inicio = fin

        return particiones

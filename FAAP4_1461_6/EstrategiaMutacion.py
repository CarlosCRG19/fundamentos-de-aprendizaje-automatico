import random as np
from abc import ABC, abstractmethod
from typing import List

from Individuo import Individuo


class EstrategiaMutacion(ABC):
    @abstractmethod
    def aplicar_mutacion(
        self, individuos: List[Individuo], probabilidad_mutacion: float
    ) -> List[Individuo]:
        pass


class MutacionEstandar(EstrategiaMutacion):
    def aplicar_mutacion(
        self, individuos: List[Individuo], probabilidad_mutacion: float
    ) -> List[Individuo]:
        descendientes = []

        for individuo in individuos:
            mutado = individuo.copia()
            punto_mutacion = np.random.randint(len(mutado.reglas))

            for i in range(len(mutado.reglas[punto_mutacion])):
                if np.random.rand() < probabilidad_mutacion:
                    mutado.reglas[punto_mutacion, i] ^= 1

            descendientes.append(mutado)

        return descendientes

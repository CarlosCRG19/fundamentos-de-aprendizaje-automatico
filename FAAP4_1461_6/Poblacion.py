from typing import List, Tuple

import numpy as np
from Individuo import Individuo


class Poblacion:
    def __init__(self, individuos: List[Individuo]):
        self._individuos = individuos

        self._promedio_fitness = -1
        self._mejor_fitness = -1
        self._mejor_individuo = None

    def fitness(self, datos_codificados: np.ndarray) -> List[Tuple[Individuo, float]]:
        fitness_poblacion = []
        suma_fitness_poblacion = 0

        mejor_fitness = -1
        mejor_individuo = None

        for individuo in self._individuos:
            fitness_individuo = individuo.fitness(datos_codificados)

            fitness_poblacion.append((individuo, fitness_individuo))
            suma_fitness_poblacion += fitness_individuo

            if fitness_individuo > mejor_fitness:
                mejor_fitness = fitness_individuo
                mejor_individuo = individuo

        self._promedio_fitness = suma_fitness_poblacion / len(self._individuos)
        self._mejor_fitness = mejor_fitness
        self._mejor_individuo = mejor_individuo

        return fitness_poblacion

    def individuos(self) -> List[Individuo]:
        return self._individuos

    def mejor_individuo(self) -> Individuo:
        return self._mejor_individuo

    def mejor_fitness(self) -> float:
        return self._mejor_fitness

    def promedio_fitness(self) -> float:
        return self._promedio_fitness

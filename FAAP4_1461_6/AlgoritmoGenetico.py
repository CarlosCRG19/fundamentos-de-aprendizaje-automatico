from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from Clasificador import Clasificador


class Individuo:
    def __init__(self, reglas: np.ndarray):
        self.reglas = reglas

    @staticmethod
    def crea_con_reglas_aleatorias(max_reglas: int, longitud_reglas: int):
        while True:
            reglas = np.random.randint(
                2, size=(np.random.randint(1, max_reglas), longitud_reglas)
            )
            # Verificar si alguna fila tiene solo 0s o solo 1s
            if not np.any(np.all(reglas == 0, axis=1)) and not np.any(
                np.all(reglas == 1, axis=1)
            ):
                break

        return Individuo(reglas)

    def clasifica_dato(self, dato: np.ndarray) -> int | None:
        votos = [0, 0]

        for regla in self.reglas:
            regla_activada = True

            for dato_bit, regla_bit in zip(dato[:-1], regla[:-1]):
                if dato_bit == 1 and regla_bit != 1:
                    regla_activada = False
                    break

            if regla_activada:
                votos[regla[-1]] += 1

        if sum(votos) == 0:
            return None
        elif votos[0] > votos[1]:
            return 0
        elif votos[0] < votos[1]:
            return 1
        else:
            return -1

    def fitness(self, datos_codificados: np.ndarray) -> float:
        aciertos = 0

        for dato in datos_codificados:
            prediccion = self.clasifica_dato(dato)

            if dato[-1] == prediccion:
                aciertos += 1

        return aciertos / datos_codificados.shape[0]

    # metodo prototipo
    def copia(self):
        return Individuo(np.copy(self.reglas))


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
                mejor_fitness = mejor_fitness
                mejor_individuo = mejor_individuo

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


# Revisar con el profesor si debemos de utilizar Datos


class CodificadorBinario:
    def __init__(self, datos: pd.DataFrame):
        self._n_bits, self._codificacion = self._init_codificacion(datos)

    def _init_codificacion(
        self, datos: pd.DataFrame
    ) -> Tuple[int, Dict[str, Dict[str, List[int]]]]:
        atributos = datos.columns[:-1]
        objetivo = datos.columns[-1]

        codificacion = {}
        n_bits = 0  # cantidad de bits necesarios para codificar una muestra

        for atributo in atributos[:-1]:
            valores = sorted(datos[atributo].astype(str).unique())
            codificacion[atributo] = {}

            for i, valor in enumerate(valores):
                codigo = [0] * len(valores)
                codigo[i] = 1

                codificacion[atributo][valor] = codigo

            n_bits += len(valores)

        # Columna de clase, tiene una codificación con un solo bit
        codificacion[objetivo] = {"0": [0], "1": [1]}
        n_bits += 1

        return n_bits, codificacion

    def codifica_datos(self, datos: pd.DataFrame) -> np.ndarray:
        filas_codificadas = []

        for _, fila in datos.iterrows():
            fila_codificada = []

            for columna, valor in fila.items():
                fila_codificada.extend(self._codificacion[columna][str(valor)])

            filas_codificadas.append(fila_codificada)

        return np.array(filas_codificadas)

    def n_bits(self) -> int:
        return self._n_bits


class AlgoritmoGenetico(Clasificador):
    def __init__(
        self,
        tamano_poblacion: int,
        n_generaciones: int,
        max_reglas: int,
        porcentaje_elitismo: float,
    ):
        self._max_reglas = max_reglas
        self._n_elitistas = int(np.ceil(tamano_poblacion * porcentaje_elitismo))
        self._n_generaciones = n_generaciones
        self._tamano_poblacion = tamano_poblacion

        self._codificador = None
        self._generaciones = []

    def entrenamiento(
        self, datosTrain: pd.DataFrame, nominalAtributos: List[bool], diccionario: Dict
    ):
        # Crea codificacion -> esto podria ser a traves de diccionario
        self._codificador = CodificadorBinario(datosTrain)

        # Crear primera generacion
        self._init_generaciones()
        datos_codificados = self._codificador.codifica_datos(datosTrain)

        for _ in range(self._n_generaciones):
            poblacion = self._generaciones[-1]

            # calcular fitness de la poblacion
            fitness_poblacion = poblacion.fitness(datos_codificados)

            # utilizar elitismo
            elite = self._selecciona_elite(fitness_poblacion)

            # selecciona progenitores
            progenitores = self._selecciona_progenitores(fitness_poblacion)

            # operadores geneticos
            descendientes = self._operador_cruce(progenitores)
            descendientes = self._operador_mutacion(descendientes)

            nuevos_individuos = elite + descendientes
            nueva_poblacion = Poblacion(nuevos_individuos)

            self._generaciones.append(nueva_poblacion)

        # seleccionar padres
        # Crossovers => cruzar los padres
        # Mutaciones => mutar los padres para generar descendientes
        # Sobrevivientes => seleccionar solo los mejores descendientes
        # actualizar poblacion

        # calcular fitness de la poblacion final
        # calcular mejor solucion (mejor individuo)

    def _init_generaciones(self) -> Poblacion:
        individuos = []

        # Generar `tamano_poblacion` individuos con reglas aleatorias
        for _ in range(self._tamano_poblacion):
            individuos.append(
                Individuo.crea_con_reglas_aleatorias(
                    max_reglas=self._max_reglas,
                    longitud_reglas=self._codificador.n_bits(),
                )
            )

        self._generaciones.append(Poblacion(individuos))

    def _selecciona_elite(
        self, fitness_poblacion: List[Tuple[Individuo, float]]
    ) -> List[Individuo]:
        # Ordenar la población según la aptitud (mayor aptitud primero)
        poblacion_ordenada = [
            individuo
            for individuo, _ in sorted(
                fitness_poblacion, key=lambda x: x[1], reverse=True
            )
        ]

        # Seleccionar a los mejores individuos (élite)
        elite = poblacion_ordenada[: self._n_elitistas]

        return elite

    def _selecciona_progenitores(
        self, fitness_poblacion: List[Tuple[Individuo, float]]
    ) -> List[Individuo]:
        n_progenitores = self._tamano_poblacion - self._n_elitistas
        suma_fitness_poblacion = sum(
            fitness_individuo for _, fitness_individuo in fitness_poblacion
        )

        # Normalizar la aptitud para convertirla en probabilidades
        probabilidad_seleccion = [
            fitness_individuo / suma_fitness_poblacion
            for _, fitness_individuo in fitness_poblacion
        ]

        # Utilizar np.random.choice para seleccionar progenitores
        progenitores_indices = np.random.choice(
            np.arange(len(fitness_poblacion)),
            size=n_progenitores,
            p=probabilidad_seleccion,
        )
        progenitores = [fitness_poblacion[i][0].copia() for i in progenitores_indices]

        return progenitores

    def _operador_cruce(self, individuos: List[Individuo]) -> List[Individuo]:
        # aplica cruce inter-reglas
        descendientes = []

        for _ in range(len(individuos) // 2):
            # Seleccionar dos progenitores aleatorios
            progenitor1, progenitor2 = np.random.choice(
                individuos, size=2, replace=False
            )

            # Realizar el cruce inter-reglas
            punto_cruce = np.random.randint(
                min(len(progenitor1.reglas), len(progenitor2.reglas))
            )

            nueva_reglas1 = np.vstack(
                (progenitor1.reglas[:punto_cruce], progenitor2.reglas[punto_cruce:])
            )
            nueva_reglas2 = np.vstack(
                (progenitor2.reglas[:punto_cruce], progenitor1.reglas[punto_cruce:])
            )

            descendiente1 = Individuo(nueva_reglas1)
            descendiente2 = Individuo(nueva_reglas2)

            descendientes.extend([descendiente1, descendiente2])

        return descendientes

    def _operador_mutacion(self, individuos: List[Individuo]) -> List[Individuo]:
        descendientes = []

        probabilidad_mutacion = 1 / (
            self._tamano_poblacion * self._codificador.n_bits()
        )

        for individuo in individuos:
            mutado = individuo.copia()
            punto_mutacion = np.random.randint(len(mutado.reglas))

            for i in range(len(mutado.reglas[punto_mutacion])):
                if np.random.rand() < probabilidad_mutacion:
                    mutado.reglas[punto_mutacion, i] ^= 1

            descendientes.append(mutado)

        return descendientes

    def _representacion_condicional(self, individuo: Individuo):
        pass

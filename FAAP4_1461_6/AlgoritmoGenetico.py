from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from CodificadorBinario import CodificadorBinario
from EstrategiaCruce import EstrategiaCruce
from EstrategiaMutacion import EstrategiaMutacion
from Individuo import Individuo
from Poblacion import Poblacion

from Clasificador import Clasificador


class AlgoritmoGenetico(Clasificador):
    def __init__(
        self,
        tamano_poblacion: int,
        n_generaciones: int,
        max_reglas: int,
        porcentaje_elitismo: float,
        probabilidad_mutacion: float,
        cruces: List[EstrategiaCruce],
        mutaciones: List[EstrategiaMutacion],
        verbose: bool,
    ):
        self._max_reglas = max_reglas
        self._n_elitistas = int(np.ceil(tamano_poblacion * porcentaje_elitismo))
        self._n_generaciones = n_generaciones
        self._tamano_poblacion = tamano_poblacion
        self._cruces = cruces
        self._mutaciones = mutaciones
        self._probabilidad_mutacion = probabilidad_mutacion

        self._verbose = verbose

        self._clase_mayoritaria = 1
        self._codificador = None
        self._generaciones = []
        self._mejor_generacion = None

    def entrenamiento(
        self, datosTrain: pd.DataFrame, nominalAtributos: List[bool], diccionario: Dict
    ):
        # Crea codificacion -> esto podria ser a traves de diccionario
        self._codificador = CodificadorBinario(datosTrain)

        datos_codificados = self._codificador.codifica_datos(datosTrain)

        # Crear primera generacion
        self._init_generaciones()
        self._init_clase_mayoritaria()

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

            if self._verbose:
                print(
                    f"Nueva poblacion, mejor: {poblacion.mejor_fitness()}, promedio: {poblacion.promedio_fitness()}"
                )

            self._generaciones.append(nueva_poblacion)

        self._generaciones[-1].fitness(datos_codificados)
        self._encuentra_mejor_generacion(datos_codificados)

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

    def _init_clase_mayoritaria(self, datos_codificados: np.ndarray):
        objetivo = datos_codificados[:, -1]
        frecuencias = np.bincount(objetivo)

        self._clase_mayoritaria = np.argmax(frecuencias)

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
        descendientes = individuos

        for cruce in self._cruces:
            descendientes = cruce.aplicar_cruce(
                descendientes, max_reglas=self._max_reglas
            )

        return descendientes

    def _operador_mutacion(self, individuos: List[Individuo]) -> List[Individuo]:
        descendientes = individuos

        for mutacion in self._mutaciones:
            descendientes = mutacion.aplicar_mutacion(
                descendientes, probabilidad_mutacion=self._probabilidad_mutacion
            )

        return descendientes

    def _encuentra_mejor_generacion(self, datos_codificados: np.ndarray):
        mejor_fitness = -1
        mejor_generacion = None

        for generacion in self._generaciones:
            mejor_individuo = generacion.mejor_individuo()
            fitness_individuo = mejor_individuo.fitness(datos_codificados)

            if fitness_individuo > mejor_fitness:
                mejor_fitness = fitness_individuo
                mejor_generacion = generacion

        self._mejor_generacion = mejor_generacion

    def representacion_condicional(self, individuo: Individuo) -> List[str]:
        reglas = individuo.reglas
        representaciones = [
            self._interpretar_regla(regla, self._codificador.codificacion())
            for regla in reglas
        ]

        return representaciones

    def _interpretar_regla(self, regla_binaria: List[int], codificacion) -> str:
        output = "\nIF "
        posicion_actual = 0

        for atributo, valores in codificacion.items():
            condiciones = [
                f"({atributo} == {list(valores.keys())[i_bit]})"
                for i_bit, bit in enumerate(
                    regla_binaria[posicion_actual : posicion_actual + len(valores)]
                )
                if bit == 1
            ]

            if condiciones:
                output += "\t AND " if output != "\nIF " else "\t"
                output += " OR ".join(condiciones)
                output += "\n"

            posicion_actual += len(valores)

        output += f"THEN Class == {regla_binaria[-1]}"

        return output

    def mejor_generacion(self):
        return self._mejor_generacion

    def clasifica(
        self, datosTest: pd.DataFrame, nominalAtributos: List[bool], diccionario: Dict
    ):
        predicciones = []
        datos_test_codificados = self._codificador.codifica_datos(datosTest)

        X_test = datos_test_codificados[:, :-1]

        for dato in X_test:
            prediccion = self._mejor_generacion.mejor_individuo().clasifica(dato)

            if prediccion == -1:
                # el modelo no pudo decidir
                prediccion = self._clase_mayoritaria

            predicciones.append(prediccion)

        return np.array(predicciones)

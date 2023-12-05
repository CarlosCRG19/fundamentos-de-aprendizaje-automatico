from random import choices
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


# Revisar con el profesor si debemos de utilizar Datos


class CodificadorBinario:
    def __init__(self, datos: pd.DataFrame):
        self._n_bits, self._codificacion = self._init_codificacion(datos)

    def _init_codificacion(
        self, datos: pd.DataFrame
    ) -> Tuple[int, Dict[str, Dict[str, List[int]]]]:
        columnas = datos.columns
        codificacion = {}
        n_bits = 0

        for columna in columnas[:-1]:
            valores = sorted(datos[columna].astype(str).unique())
            codificacion[columna] = {}

            for i, valor in enumerate(valores):
                codigo = [0] * len(valores)
                codigo[i] = 1

                codificacion[columna][valor] = codigo

            n_bits += len(valores)

        # Columna de clase, tiene una codificación con un solo bit
        codificacion[columnas[-1]] = {"0": [0], "1": [1]}
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
    def __init__(self, tamano_poblacion: int, epocas: int, max_reglas: int):
        self.tamano_poblacion = tamano_poblacion
        self.epocas = epocas
        self.max_reglas = max_reglas
        self.poblacion = []

    def entrenamiento(
        self, datosTrain: pd.DataFrame, nominalAtributos: List[bool], diccionario: Dict
    ):
        # Crea codificacion -> esto podria ser a traves de diccionario
        self.codificador = CodificadorBinario(datosTrain)
        # Crear primera generacion
        self.poblacion = self._crea_primera_generacion()

        datos_codificados = self.codificador.codifica_datos(datosTrain)

        # for _ in range(epocas):
        # calcular fitness de la poblacion
        # utilizar elitismo
        # seleccionar padres
        # Crossovers => cruzar los padres
        # Mutaciones => mutar los padres para generar descendientes
        # Sobrevivientes => seleccionar solo los mejores descendientes
        # actualizar poblacion

        # calcular fitness de la poblacion final
        # calcular mejor solucion (mejor individuo)

    def _crea_primera_generacion(self, datosTrain: pd.DataFrame) -> List[Individuo]:
        poblacion = []

        # Generar `tamano_poblacion` individuos con reglas aleatorias
        for _ in range(self.tamano_poblacion):
            poblacion.append(
                Individuo.crea_con_reglas_aleatorias(
                    max_reglas=self.max_reglas,
                    longitud_reglas=self.codificador.n_bits(),
                )
            )

        return poblacion

    # Esta parte necesesita la codificacion terminada
    def _seleccion_progenitores(
        self, datos: pd.DataFrame, individuos: List[Individuo]
    ) -> List[float]:
        lista_fitness = [
            self._evalua_fitness(datos, individuo) for individuo in individuos
        ]
        ruleta = [fitness / sum(lista_fitness) for fitness in lista_fitness]

        cantidad_progenitores = round(0.8 * len(individuos))
        seleccion = choices(individuos, weights=ruleta, k=cantidad_progenitores)
        return seleccion

    def _evalua_fitness(self, datos: pd.DataFrame, individuo: Individuo) -> float:
        clasificaciones = []

        # clasifica todos los datos con individuo
        for _, dato in datos.iterrows():
            # Llama _fitness_regla con cada regla y tiene cuenta de las clasificaciones correctas
            cada_regla = [
                True if self._fitness_regla(dato, regla) else False
                for regla in individuo.reglas
            ]

            # Si la mayoria de reglas clasifican bien, el dato ha sido clasificado correctamente
            if sum(cada_regla) > np.floor(len(cada_regla) / 2):
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
            sum(cada_atributo) < len(cada_atributo)
            and dato[-1] != regla[-1]
            or all(cada_atributo)
            and dato[-1] == regla[-1]
        ):
            return True
        return False

from typing import Dict, List

import numpy as np
import pandas as pd
from CalculadoraProbabilidad import (CalculadoraProbabilidad,
                                     ProbabilidadGaussiana,
                                     ProbabilidadNominal,
                                     ProbabilidadNominalConLaplace)

from Clasificador import Clasificador


class ClasificadorNaiveBayes(Clasificador):
    def __init__(self, con_laplace=False):
        self.con_laplace = con_laplace

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        self._columnas_atributos = datosTrain.columns[:-1]
        self._columna_clase = datosTrain.columns[-1]

        # Estructuras de datos utilizadas en para las predicciones de nuevos conjuntos
        # de datos. Se utilizan en la ecuación del teorema de Bayes
        # P(y|x1,...,xj) = (P(x1,...,xj|y) * P(y)) / P(x1,...,xj)

        self._a_priori = self._inicializa_a_priori(datosTrain)  # P(y)
        self._evidencias = self._inicializa_evidencias(
            datosTrain, nominalAtributos
        )  # P(x1,...,xj)
        self._verosimilitudes = self._inicializa_verosimilitudes(
            datosTrain, nominalAtributos
        )  # P(x1,...,xj | y)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        predicciones = []
        clases = datosTest[self._columna_clase].unique()

        X = datosTest.iloc[:, :-1]  # se remueve la ultima columna, "Clase"

        # para cada registro, se genera una prediccion
        for _, fila in X.iterrows():
            probabilidades_posteriores = {}

            for clase in clases:
                probabilidades_posteriores[
                    clase
                ] = self._calcula_probabilidad_a_posteriori(fila, clase)

            # se toma la clase con la probabilidad más alta
            clase_predicha = max(
                probabilidades_posteriores, key=probabilidades_posteriores.get
            )
            predicciones.append(clase_predicha)

        return np.array(predicciones)

    def _inicializa_evidencias(
        self, datos: pd.DataFrame, nominalAtributos: List[bool]
    ) -> Dict[str, CalculadoraProbabilidad]:
        """
        inicializa las estructuras de evidencia para cada atributo.
        la evidencia representa la probabilidad de observar datos para un atributo
        sin tener en cuenta una clase específica.

        Argumentos:
            - datos: el conjunto de datos de entrenamiento.
            - nominalatributos: una lista que indica si la columna en la posición i es nominal o continua.

        Retorna:
            - un diccionario donde las claves son los atributos del conjunto de datos y los valores son
              objetos que calculan la probabilidad de un valor para ese atributo.

              e.g.
                    {
                      "AtributoA": CalculadoraProbabilidad(),
                      "AtributoB": CalculadoraProbabilidad()
                    }
        """
        evidencias = {}

        for atributo in self._columnas_atributos:
            evidencias[atributo] = self._crea_calculadora_probabilidad(
                datos, atributo, nominalAtributos
            )

        return evidencias

    def _inicializa_a_priori(self, datos: pd.DataFrame):
        """
        Inicializa un objeto que conoce las probabilidades de ocurrencia de cada clase.
        Distingue si se debe de usar la correccion de Laplcase o no.

        Argumentos:
            - datos: El conjunto de datos de entrenamiento.

        Retorna:
            - un objeto que conoce las probabilidades de ocurrencia de cada clase.
        """
        if self.con_laplace:
            return ProbabilidadNominalConLaplace(datos[self._columna_clase])
        else:
            return ProbabilidadNominal(datos[self._columna_clase])

    def _inicializa_verosimilitudes(
        self, datos: pd.DataFrame, nominalAtributos: List[bool]
    ) -> Dict[str, Dict[str, CalculadoraProbabilidad]]:
        """
        inicializa las estructuras de verosimilitued para cada atributo y clase.
        La verosimilitud (likelihood) representa la probabilidad de observar datos para un atributo
        tomando en cuenta una clase específica.

        Argumentos:
            - datos: el conjunto de datos de entrenamiento.
            - nominalatributos: una lista que indica si la columna en la posición i es nominal o continua.

        Retorna:
            - un diccionario multinivel donde las primeras claves son los atributos del conjunto de datos;
              las del segundo, las clases posibles, y los valores son objetos que calculan
              la probabilidad de un valor para ese atributo dada una clase.

              e.g.
                    {
                      "AtributoA": {
                          "ClaseX": CalculadoraProbabilidad(),
                          "ClaseY": CalculadoraProbabilidad(),
                      },
                      "AtributoB": {
                          "ClaseX": CalculadoraProbabilidad(),
                          "ClaseY": CalculadoraProbabilidad(),
                      }
                    }
        """
        verosimilitudes = {atributo: {} for atributo in self._columnas_atributos}

        for clase, datos_clase in datos.groupby(self._columna_clase):
            for atributo in self._columnas_atributos:
                verosimilitudes[atributo][clase] = self._crea_calculadora_probabilidad(
                    datos_clase, atributo, nominalAtributos
                )

        return verosimilitudes

    def _calcula_probabilidad_a_posteriori(self, fila: pd.Series, clase: int):
        a_priori = self._a_priori.calcula_para_valor(clase)
        evidencia = 1
        verosimilitud = 1

        for atributo in self._columnas_atributos:
            valor_atributo = fila[atributo]

            # Productorios
            evidencia *= self._evidencias[atributo].calcula_para_valor(valor_atributo)
            verosimilitud *= self._verosimilitudes[atributo][clase].calcula_para_valor(
                valor_atributo
            )

        # (P(x1,...,xj|y) * P(y)) / P(x1,...,xj)
        return (a_priori * verosimilitud) / evidencia

    def _crea_calculadora_probabilidad(self, datos, atributo, nominalAtributos):
        """
        Método creacional de instancias de CalculadoraProbabilidad, dependiendo
        del atributo que se quiere evaluar. Distingue si el atributo es nominal o numerico
        y si se debe de usar la corrección de Laplace

        Argumentos:
            - datos: el conjunto de datos de entrenamiento.
            - atributo: el nombre de un atributo en nuestro dataset.
            - nominalatributos: una lista que indica si la columna en la posición i es nominal o continua.

        Retorna:
            - una instancia de una subclase de CalculadoraProbabilidad que conoce la probabilidad de un valor
              para el atributo especificado
        """
        es_nominal = nominalAtributos[datos.columns.get_loc(atributo)]

        if es_nominal:
            if self.con_laplace:
                calculadora_probabilidad = ProbabilidadNominalConLaplace(
                    datos[atributo]
                )
            else:
                calculadora_probabilidad = ProbabilidadNominal(datos[atributo])
        else:
            calculadora_probabilidad = ProbabilidadGaussiana(datos[atributo])

        return calculadora_probabilidad

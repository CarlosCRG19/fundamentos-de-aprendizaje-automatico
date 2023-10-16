from abc import ABCMeta, abstractmethod
from typing import Dict, List, NewType

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

PandasSeries = NewType("PandasSeries", pd.core.series.Series)


class CalculadoraProbabilidad:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calcula_para_valor(self, valor) -> float:
        pass


class ProbabilidadNominal(CalculadoraProbabilidad):
    def __init__(self, valores_nominales: PandasSeries):
        conteo_de_valores = valores_nominales.value_counts()
        probabilidades = conteo_de_valores / len(valores_nominales)
        self._probabilidades = probabilidades.to_dict()

    def calcula_para_valor(self, valor) -> float:
        if valor in self._probabilidades:
            return self._probabilidades[valor]
        else:
            return 0


class ProbabilidadGaussiana(CalculadoraProbabilidad):
    def __init__(self, valores_numericos: PandasSeries):
        self.media = valores_numericos.mean()
        self.desviacion_estandar = valores_numericos.std()

    def calcula_para_valor(self, valor) -> float:
        if self.desviacion_estandar == 0:
            return 1.0

        exponente = -((valor - self.media) ** 2) / (2 * self.desviacion_estandar**2)
        denominador = self.desviacion_estandar * np.sqrt(2 * np.pi)

        return (1 / denominador) * np.exp(exponente)


class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto. Crea el modelo a partir de los datos de entrenamiento
    # datosTrain: matriz numpy con los datos de entrenamiento
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        pass

    @abstractmethod
    # TODO: esta funcion debe ser implementada en cada clasificador concreto. Devuelve un numpy array con las predicciones
    # datosTest: matriz numpy con los datos de validaci�n
    # nominalAtributos: array bool con la indicatriz de los atributos nominales
    # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
    def clasifica(self, datosTest, nominalAtributos, diccionario):
        pass

    # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
    def error(self, datos, pred):
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        valores_reales = datos.iloc[:, -1]
        1 - accuracy_score(valores_reales, pred)

    # Realiza una clasificacion utilizando una estrategia de particionado determinada
    def validacion(self, particionado, dataset, clasificador, seed=None):
        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        particionado.creaParticiones(dataset.datos)

        errores = []

        for particion in particionado.particiones:
            datos_test = dataset.extraeDatos(particion.indicesTest)
            datos_train = dataset.extraeDatos(particion.indicesTrain)

            clasificador.entrenamiento(
                datos_train, dataset.nominalAtributos, dataset.diccionario
            )

            predicciones = clasificador.clasifica(
                datos_test, dataset.nominalAtributos, dataset.diccionario
            )

            error = self.error(datos_test, predicciones)
            errores.append(error)

        return np.mean(errores)


##############################################################################


class ClasificadorNaiveBayes(Clasificador):
    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        self._columnas_atributos = datosTrain.columns[:-1]
        self._columna_clase = datosTrain.columns[-1]

        self._a_priori = ProbabilidadNominal(datosTrain[self._columna_clase])
        self._evidencias = self._inicializa_evidencias(datosTrain, nominalAtributos)
        self._verosimilitudes = self._inicializa_verosimilitudes(
            datosTrain, nominalAtributos
        )

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        predicciones = []

        for _, fila in datosTest.iterrows():
            probabilidades_posteriores = {}

            for clase in datosTest[self._columna_clase].unique():
                probabilidades_posteriores[
                    clase
                ] = self._calcula_probabilidad_a_posteriori(fila, clase)

            clase_predicha = max(
                probabilidades_posteriores, key=probabilidades_posteriores.get
            )
            predicciones.append(clase_predicha)

        return np.array(predicciones)

    def _inicializa_evidencias(
        self, datos: pd.DataFrame, nominalAtributos: List[bool]
    ) -> Dict[str, CalculadoraProbabilidad]:
        evidencias = {}
        atributos = datos.iloc[:, :-1]

        for i, columna in enumerate(atributos.columns):
            if nominalAtributos[i]:
                evidencias[columna] = ProbabilidadNominal(atributos[columna])
            else:
                evidencias[columna] = ProbabilidadGaussiana(atributos[columna])

        return evidencias

    def _inicializa_verosimilitudes(
        self, datos: pd.DataFrame, nominalAtributos: List[bool]
    ) -> Dict[str, Dict[str, CalculadoraProbabilidad]]:
        verosimilitudes = {atributo: {} for atributo in self._columnas_atributos}

        for clase, datos_clase in datos.groupby(self._columna_clase):
            for i, atributo in enumerate(self._columnas_atributos):
                if nominalAtributos[i]:
                    verosimilitudes[atributo][clase] = ProbabilidadNominal(
                        datos_clase[atributo]
                    )
                else:
                    verosimilitudes[atributo][clase] = ProbabilidadGaussiana(
                        datos_clase[atributo]
                    )

        return verosimilitudes

    def _calcula_probabilidad_a_posteriori(self, fila: PandasSeries, clase: int):
        a_priori = self._a_priori.calcula_para_valor(clase)
        evidencia = 1
        verosimilitud = 1

        for atributo in self._columnas_atributos:
            valor_atributo = fila[atributo]

            evidencia *= self._evidencias[atributo].calcula_para_valor(valor_atributo)
            verosimilitud *= self._verosimilitudes[atributo][clase].calcula_para_valor(
                valor_atributo
            )

        return (a_priori * verosimilitud) / evidencia

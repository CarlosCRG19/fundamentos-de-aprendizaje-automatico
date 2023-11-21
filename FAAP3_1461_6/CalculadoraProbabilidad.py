from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

# Clase auxiliar para calcular la probabilidad de un valor para un atributos
# especifico del dataset


class CalculadoraProbabilidad:
    __metaclass__ = ABCMeta

    @abstractmethod
    def calcula_para_valor(self, valor) -> float:
        pass


class ProbabilidadNominal(CalculadoraProbabilidad):
    def __init__(self, valores_nominales: pd.Series):
        conteo_de_valores = valores_nominales.value_counts()
        probabilidades = conteo_de_valores / len(valores_nominales)
        self._probabilidades = probabilidades.to_dict()

    def calcula_para_valor(self, valor) -> float:
        if valor in self._probabilidades:
            return self._probabilidades[valor]
        else:
            return 0


class ProbabilidadNominalConLaplace(CalculadoraProbabilidad):
    def __init__(self, valores_nominales: pd.Series, correccion=1):
        self.conteo_de_valores = valores_nominales.value_counts()
        self.n = len(valores_nominales)
        self.k = len(self.conteo_de_valores)
        self.correccion = 1

    def calcula_para_valor(self, valor) -> float:
        if valor in self.conteo_de_valores:
            return (self.conteo_de_valores[valor] + self.correccion) / (self.n + self.k)
        else:
            return self.correccion / (self.n + self.k)


class ProbabilidadGaussiana(CalculadoraProbabilidad):
    """
    Calcula probabilidades para valores continuos siguiendo una distribución Gaussiana
    """

    def __init__(self, valores_numericos: pd.Series):
        self.media = valores_numericos.mean()
        self.desviacion_estandar = valores_numericos.std()

    def calcula_para_valor(self, valor) -> float:
        # retorna un valor constante si la desviacion estandar es 0
        # basicamente hace que se ignore la columna para el calculo de la prediccion
        if self.desviacion_estandar == 0:
            return 1.0

        exponente = -((valor - self.media) ** 2) / (2 * self.desviacion_estandar**2)
        denominador = self.desviacion_estandar * np.sqrt(2 * np.pi)

        return (1 / denominador) * np.exp(exponente)

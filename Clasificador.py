from abc import ABCMeta, abstractmethod
from typing import Dict, List, NewType, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import CategoricalNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import OneHotEncoder

# Tipo personalizado para Pandas Series
PandasSeries = NewType("PandasSeries", pd.core.series.Series)


##############################################################################


# Clase auxiliar para calcular la probabilidad de un valor para un atributos
# especifico del dataset
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


class ProbabilidadNominalConLaplace(CalculadoraProbabilidad):
    def __init__(self, valores_nominales: PandasSeries, correccion=1):
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

    def __init__(self, valores_numericos: PandasSeries):
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


##############################################################################


class Clasificador:
    # Clase abstracta
    __metaclass__ = ABCMeta

    # Metodos abstractos que se implementan en casa clasificador concreto
    @abstractmethod
    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        """
        Ajusta el modelo al conjunto de entrenamiento, generando los valores
        necesarios para clasificar nuevos datos.

        Argumentos:
            - datosTrain: el conjunto de datos de entrenamiento
            - nominalAtributos: array que indica si la columna en la posicion i es nominal o continua
            - diccionario: valores de mapeo de las variables categoricas a numeros
        """
        pass

    @abstractmethod
    def clasifica(self, datosTest, nominalAtributos, diccionario):
        """
        Utiliza el modelo ajustado para realizar predicciones sobre un nuevo conjunto
        de datos.

        Argumentos:
            - datosTrain: el conjunto de datos de entrenamiento
            - nominalAtributos: array que indica si la columna en la posicion i es nominal o continua
            - diccionario: valores de mapeo de las variables categoricas a numeros

        Retorna:
            - una lista de longitud len(datosTest) con la prediccion de la clase de cada registro/fila
        """
        pass

    def error(self, datos, pred):
        """
        Obtiene el numero de aciertos y errores para calcular la tasa de fallo
        """
        # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error
        valores_reales = datos.iloc[:, -1]
        return 1 - accuracy_score(valores_reales, pred)

    def validacion(self, particionado, dataset, clasificador, seed=None):
        """
        Realiza una clasificacion utilizando una estrategia de particionado determinada
        """
        errores = []

        # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
        particionado.creaParticiones(dataset.datos)

        for particion in particionado.particiones:
            datos_test = dataset.extraeDatos(particion.indicesTest)
            datos_train = dataset.extraeDatos(particion.indicesTrain)

            # se llama al metodo del clasificador para ajustar el modelo al conjunto
            # de entrenamiento
            clasificador.entrenamiento(
                datos_train, dataset.nominalAtributos, dataset.diccionarios
            )

            # obtenemos la lista de clases predichas para cada registro
            predicciones = clasificador.clasifica(
                datos_test, dataset.nominalAtributos, dataset.diccionarios
            )

            error = self.error(datos_test, predicciones)
            errores.append(error)

        return np.mean(errores), np.std(errores)


##############################################################################
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

    def _calcula_probabilidad_a_posteriori(self, fila: PandasSeries, clase: int):
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


##############################################################################
class ClasificadorNaiveBayesScikit:
    def __init__(self, modelo: Union[CategoricalNB, GaussianNB, MultinomialNB]):
        self._modelo = modelo
        self._codificador = OneHotEncoder()

    def entrenamiento(self, datosTrain, nominalAtributos, diccionario):
        """
        Entrena el clasificador Naive Bayes utilizando los datos de entrenamiento.

        Args:
            datosTrain (pd.DataFrame): El conjunto de datos de entrenamiento.
            nominalAtributos (List[bool]): Lista de booleanos indicando si los atributos son nominales o no.
            diccionario: No se utiliza en este método.
        """
        X = datosTrain.iloc[:, :-1]
        y = datosTrain.iloc[:, -1]

        # Codificacion de atributos nominales. Se utiliza One Hot Encoding,
        # para que no se interpreten las categorias como valores numericos sucesivos
        columnas_atributos_nominales = [
            columna for i, columna in enumerate(X.columns) if nominalAtributos[i]
        ]
        self._transformador = ColumnTransformer(
            transformers=[("nominal", self._codificador, columnas_atributos_nominales)],
            remainder="passthrough",  # columnas no nominales
        )
        X_codificado = self._transformador.fit_transform(X)

        self._modelo.fit(X_codificado, y)

    def clasifica(self, datosTest, nominalAtributos, diccionario):
        X = datosTest.iloc[:, :-1]
        # se utiliza el transformador ajustado con el dataset de entrenamiento
        # para reemplazar las columnas categoricas
        X_codificado = self._transformador.transform(X)

        return self._modelo.predict(X_codificado)

from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score

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
        particionado.creaParticiones(dataset.datos, seed=55)

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
